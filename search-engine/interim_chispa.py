#!/usr/bin/python3
"""
Create, update, or query a bare-bones full-text index.
"""

debug = 1

import gzip
import heapq                    # for merging
import itertools
import os
from os.path import abspath, basename, join, relpath
import pickle
import re                       # to extract words from input
import shutil                   # to remove directory trees
import sys
import time
try:                import urllib.parse as quoting # for quote and unquote (Py3)
except ImportError: import urllib as quoting       # (Py2)

def main(argv):
    # Eliminate the most common English words from queries and indices.
    stopwords = 'the of and to a in it is was that i for on you he be'.split()
    omit_stopwords = make_stopwords_filter(stopwords)
    transform = compound_transform(omit_long_nonsense_words,
                                   omit_stopwords,
                                   also_lowercased)

    if argv[1] == 'new':
        write_empty_index(argv[2])
    elif argv[1] == 'index':
        ask_index(argv[2], 'update', corpus_path=argv[3], term_transform=transform)
    elif argv[1] == 'find':
        search_ui(argv[2], argv[3:], omit_stopwords)
    else:
        raise Exception("%s (new|index|find) index_dir ..." % argv[0])

def search_ui(index_path, query, term_filter):
    terms = set(term_filter(query))
    if not terms:
        raise Exception("The query is empty after stripping stopwords")
    paths = search(index_path, terms)
    # Use the crudest possible ranking: newest (largest mtime) first.
    for path in sorted(paths): # XXX temp for deterministic testing
#                       key=get_metadata, reverse=True):
        print(path)

def search(index_path, terms):
    """Yield the paths, relative to ., of all the files that, according to
    the index, have all of the terms."""
    for chances in range(4, -1, -1):
        try:
            paths = list(ask_index(index_path, 'search', query=terms))
        except IOError:
            # We're presuming an I/O error is due to a concurrent writer
            # deleting an index file we needed, unless it keeps happening.
            # XXX find a more-selective kind of IOError?
            if chances == 0: raise
    for path in paths:
        yield relpath(path, start='.')


# Term-stream transforms

def compound_transform(*transforms):
    def compound(terms):
        for transform in transforms:
            terms = transform(terms)
        return terms
    return compound

def make_stopwords_filter(stopwords):
    stopwords = set(  stopwords
                    + [word.upper()      for word in stopwords]
                    + [word.capitalize() for word in stopwords])
    return lambda terms: (term for term in terms if term not in stopwords)

def omit_long_nonsense_words(terms):
    return (term for term in terms if len(term) < nonsense_len)

nonsense_len = 20

def also_lowercased(terms):
    """Let searches for a term in lowercase find also the instances of the
    same term in other cases."""
    for term in terms:
        yield term
        if term.lower() != term:
            yield term.lower()


# An index is a directory holding runs (each also a directory; a
# term's info may be in any of the runs). It also holds a
# 'catalog.pickle' file, which lists the runs and their lengths, plus
# the indexed documents (mapping numeric ids to file paths, along with
# their metadata). A document's id changes when it gets reindexed,
# which happens when its metadata in the filesystem doesn't match the
# catalog. (The id-renaming lets us defer updating big datafiles; the
# catalog is ordinarily much smaller.) Finally, the catalog stores the
# next available doc-id.

# Document paths are relative to the index's parent directory.

# A run represents a sorted sequence of postings, where a posting
# gives a term and a doc-id.

# The run is represented by a span-map file and corresponding
# compressed span files. Each span is a contiguous segment of the run.
# The span-map lists them in term order, one line for each span,
# giving its first and last terms.

def write_empty_index(path):
    os.mkdir(path)
    with open(join(path, 'catalog.pickle'), 'wb') as f:
        pickle.dump({'runs': {}, 'docs_by_path': {}, 'next_id': 0}, f)

def ask_index(index_path, command, **arguments):
    "Open the index at `index_path`, and perform `command` on it."

    if command == 'update':
        # Start a write transaction.
        try:
            write_lock = open(join(index_path, 'write_lock'), 'xb')
        except FileExistsError:
            raise Exception("The index at %s seems to be busy on another update."
                            % index_path)

    catalog_path = join(index_path, 'catalog.pickle')
    with open(catalog_path, 'rb') as catalog_file:
        catalog = pickle.load(catalog_file)

    runs = {join(index_path, str(run_id)): size
            for run_id, size in catalog['runs'].items()}
    next_runs = itertools.count(max(catalog['runs']) + 1 if runs else 0)
    def new_run():
        return join(index_path, str(next(next_runs)))

    trash = []                  # Directories to delete

    docs_by_path = catalog['docs_by_path']
    paths_by_doc_id = {doc_id: path
                       for path, (doc_id, _) in docs_by_path.items()}

    next_doc_ids = (str(ii) for ii in itertools.count(int(catalog['next_id'])))

    def write_catalog(catalog_file):
        pickle.dump({'runs': {int(basename(run)): size
                              for run, size in runs.items()},
                     'docs_by_path': docs_by_path,
                     'next_id': next(next_doc_ids)},
                    catalog_file)

    index_parent = os.path.dirname(index_path)
    def to_relpath(path):   return relpath(path, start=index_parent)
    def from_relpath(path): return join(index_parent, path)

    def search(query):
        "Return the paths of the documents that have all of `query`'s terms."
        return (paths_by_doc_id[doc_id]
                for doc_id in set.intersection(*(set(find(term))
                                                 for term in query)))

    def find(term):
        return itertools.chain(*(find_in_run(run, term) for run in runs))

    def update(corpus_path, term_transform):
        try:
            updates = muster_updates(corpus_path)
            postings = tokenize_documents(updates, term_transform)
            # Write all the new postings, first into runs, then merged.
            for run_postings in chunked(postings, run_size):
                write_new_run(sorted(run_postings))
            merge_some_runs()
            # Commit to the updated catalog.
            write_catalog(write_lock)
            os.rename(write_lock.name, catalog_path)
            write_lock.close()
        except: # Or roll it back.
            os.remove(write_lock.name)
            raise
        else:
            for run in trash:
                shutil.rmtree(run)

    def muster_updates(corpus_path):
        """Yield the documents under corpus_path that have changed since the
        last reindexing, updating the catalog for them as we go."""
        paths = list(muster_files(to_relpath(corpus_path)))

        # Note deleted documents.
        for catalogued in set(docs_by_path) - set(paths):
            if is_under(corpus_path, catalogued):
                del paths_by_doc_id[docs_by_path[catalogued][0]]
                del docs_by_path[catalogued]
                if debug: print('missing', basename(catalogued))

        # Note and yield new and changed documents.
        for path in paths:
            metadata = get_metadata(path)
            old_id, old_metadata = docs_by_path.get(path, (None, None))
            if old_metadata != metadata:
                if old_id is not None:
                    del paths_by_doc_id[old_id]
                doc_id = next(next_doc_ids)
                paths_by_doc_id[doc_id] = path
                docs_by_path[path] = (doc_id, metadata)
                if debug: print('indexing', basename(path))
                yield path, doc_id

    # 2**20 is chosen as the maximum run size because that uses
    # typically about a quarter gig, which is a reasonable size these
    # days.
    run_size = 10 if debug else 2**20

    def merge_some_runs():
        """Combine the smaller runs into one; but stop short of merging only a
        little data with a lot."""
        mergees, sizes = zip(*sorted(runs.items(), key=lambda item: item[1]))
        total, overflow = 0, None
        for ii, size in enumerate(sizes):
            if size <= total: overflow = ii
            total += size
        if overflow is not None:
            merger = mergees[:overflow+1]
            write_new_run(heapq.merge(*map(read_run, merger)))
            trash.extend(merger)
            for run in merger:
                del runs[run]

    def read_run(run):
        for span_name, _ in muster_spans(run):
            for posting in read_span(join(run, span_name)):
                yield posting

    def write_new_run(postings):
        # Pre: postings is sorted.
        run = new_run()
        os.mkdir(run)
        size = 0
        with open(join(run, 'span-map'), 'w') as map_file:
            for ii, span_postings in enumerate(chunked(postings, span_size)):
                span_postings = list(span_postings)
                with gzip.open(join(run, '%s.gz' % ii), 'wt') as span_file:
                    write_tuples(span_file, span_postings)
                write_tuple(map_file, (span_postings[-1][0],))
                size += len(span_postings)
        runs[run] = size

    span_size = 3 if debug else 4096

    def find_in_run(run, needle_term):
        "Yield a run's doc_ids that have needle_term."
        for span in find_spans(run, needle_term):
            for haystack_term, doc_id in read_span(span):
                if haystack_term == needle_term:
                    yield doc_id
                # Once we reach an alphabetically later term, we're done:
                if haystack_term > needle_term:
                    break

    def read_span(span):
        with gzip.open(span, 'rt') as f:
            for posting in read_tuples(f):
                if posting[1] in paths_by_doc_id:
                    yield posting

    def find_spans(run, term):
        "Yield the spans within the run that may have the term."
        lo = ''
        for span_name, hi in muster_spans(run):
            if term < lo:
                break
            if term <= hi:
                yield join(run, span_name)
            lo = hi

    def muster_spans(run):
        with open(join(run, 'span-map')) as f:
            for ii, (hi,) in enumerate(read_tuples(f)):
                yield '%s.gz' % ii, hi

    if command == 'search':
        return search(**arguments)
    elif command == 'update':
        update(**arguments)
    else:
        assert False


# Tokenizing

def tokenize_documents(documents, term_transform):
    for path, doc_id in documents:
        for term in keep_unique(term_transform(keep_unique(tokenize_file(path)))):
            yield term, doc_id

def tokenize_file(file_path):
    tokenizer = tokenize_html if file_path.endswith('.html') else tokenize_text
    with open(file_path) as fo:
        for term in tokenizer(fo):
            yield term

def tokenize_text(fo):
    word_re = re.compile(r'\w+')
    for line in fo:
        for word in word_re.findall(line):
            yield word

# Crude approximation of HTML tokenization.  Note that for proper
# excerpt generation (as in the "grep" command) the postings generated
# need to contain position information, because we need to run this
# tokenizer during excerpt generation too.
def tokenize_html(fo):
    tag_re       = re.compile('<.*?>')
    tag_start_re = re.compile('<.*')
    tag_end_re   = re.compile('.*?>')
    word_re      = re.compile(r'\w+')

    in_tag = False
    for line in fo:

        if in_tag and tag_end_re.search(line):
            line = tag_end_re.sub('', line)
            in_tag = False

        elif not in_tag:
            line = tag_re.subn('', line)[0]
            if tag_start_re.search(line):
                in_tag = True
                line = tag_start_re.sub('', line)
            for term in word_re.findall(line):
                yield term


# Tuple encoding

def write_tuples(outfile, tuples):
    for item in tuples:
        write_tuple(outfile, item)

def write_tuple(outfile, item):
    outfile.write(' '.join(map(quoting.quote, item)) + '\n')

def read_tuples(infile):
    for line in infile:
        yield tuple(map(quoting.unquote, line.split()))


# Helpers

def muster_files(path):
    "Yield all file paths in the tree under `path`."
    for dir_path, _, filenames in os.walk(path):
        for filename in filenames:
            yield join(dir_path, filename)

# From nikipore on Stack Overflow <http://stackoverflow.com/a/19264525>
def chunked(iterable, chunk_size):
    it = iter(iterable)
    while True:
        # XXX for some reason using list(), and then later sorting in
        # place, makes the whole program run twice as slow and doesn't
        # reduce its memory usage.  No idea why.
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk: break
        yield chunk

def keep_unique(iterable):
    "Yield only the items from iterable that are different."
    seen_items = set()
    for item in iterable:
        if item not in seen_items:
            yield item
            seen_items.add(item)

def get_metadata(path):
    "Return the modification-time and size of a file."
    s = os.stat(path)
    return int(s.st_mtime), int(s.st_size)

def is_under(dir_path, file_path):
    """Return true if the path `file_path` is part of the filesystem
    subtree headed by `dir_path`."""
    # TODO surely there's a better way?
    return abspath(file_path).startswith(abspath(dir_path) + os.sep)


if __name__ == '__main__':
    main(sys.argv)
