#!/usr/bin/python3
"""
Create, update, or query a bare-bones full-text index.
"""

from __future__ import division

debug = 1

from collections import Counter, defaultdict
from contextlib import contextmanager
from math import log
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
        with open_for_update(argv[2]) as index:
            index.update(corpus_path=argv[3], term_transform=transform)
    elif argv[1] == 'find':
        search_ui(argv[2], argv[3:], omit_stopwords)
    else:
        raise Exception("%s (new|index|find) index_dir ..." % argv[0])

def search_ui(index_path, query, term_filter, relfind=False):
    terms = set(term_filter(query))
    if not terms:
        sys.stderr.write("The query is empty after stripping stopwords\n")
        sys.exit(1) # XXX was raise, but I don't want a traceback in my tush script
    for relevance, path in relsearch(index_path, terms):
        print('%3d %s' % (relevance*1000, path))

def relsearch(index_path, terms):
    for chances in range(4, -1, -1):
        try:
            with open_for_reading(index_path) as index:
                results = list(index.search(query=terms))
        except IOError:
            # We're presuming an I/O error is due to a concurrent writer
            # deleting an index file we needed, unless it keeps happening.
            # XXX find a more-selective kind of IOError?
            if chances == 0: raise
    for relevance, path in results:
        yield relevance, relpath(path, start='.')


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
# their metadata and their lengths in words). A document's id changes
# when it gets reindexed, which happens when its metadata in the
# filesystem doesn't match the catalog. (The id-renaming lets us defer
# updating big datafiles; the catalog is ordinarily much smaller.)
# Finally, the catalog stores the next available doc-id (needed
# because the runs might contain old doc-ids higher than any still in
# the documents table.)

# Document paths are relative to the index's parent directory.

# A run represents a sorted sequence of postings, where a posting
# gives a term and a doc-id.

# The run is represented by a span-map file and corresponding
# compressed span files. Each span is a contiguous segment of the run.
# The span-map lists them in term order, one line for each span,
# telling the span's last term.

def write_empty_index(path):
    os.mkdir(path)
    with open(join(path, 'catalog.pickle'), 'wb') as f:
        pickle.dump({'runs': {}, 'documents': {}, 'next_id': 0}, f)

# TODO: Also, could I move the retry-on-IO-error to here somehow?
@contextmanager
def open_for_reading(path):
    yield Index(path)

class Index(object):
    def __init__(self, path):
        self.my_path = path
        self.parent_path = os.path.dirname(path)
        self.catalog_path = join(path, 'catalog.pickle')
        with open(self.catalog_path, 'rb') as catalog_file:
            catalog = pickle.load(catalog_file)
        self.runs = {join(path, str(run_id)): size
                     for run_id, size in catalog['runs'].items()}
        self.documents = catalog['documents']
        self.doc_ids = {path: doc_id for doc_id, (path, _, _) in self.documents.items()}
        # XXX These are used only by WritableIndex:
        self.next_doc_ids = (str(ii) for ii in itertools.count(int(catalog['next_id'])))
        self.next_runs = itertools.count(max(catalog['runs']) + 1 if self.runs
                                         else 0)

    def doc_path(self, doc_id):     return self.from_relpath(self.documents[doc_id][0])
    def doc_length(self, doc_id):   return self.documents[doc_id][1]
    def doc_metadata(self, doc_id): return self.documents[doc_id][2]

    def to_relpath(self, path):   return relpath(path, start=self.parent_path)
    def from_relpath(self, path): return join(self.parent_path, path)

    def search(self, query):
        """Return the scores and paths of the documents that have any of
        `query`'s terms, higher scores first."""
        return ((relevance, self.doc_path(doc_id))
                for relevance, doc_id in sorted(self.unordered_search(query),
                                                reverse=True))

    def unordered_search(self, terms):
        scores = defaultdict(float)
        for term in terms:
            postings = list(self.find(term))
            scale = log(1 + len(self.documents) / len(postings)) if postings else 0
            for _, doc_id, count in postings:
                scores[doc_id] += int(count) * scale
        return ((score/self.doc_length(doc_id), doc_id)
                for doc_id, score in scores.items())    

    def find(self, term):
        for run in self.runs:
            for posting in self.find_in_run(run, term):
                yield posting

    def read_run(self, run):
        for span_name, _ in self.muster_spans(run):
            for posting in self.read_span(join(run, span_name)):
                yield posting

    def find_in_run(self, run, needle_term):
        "Yield a run's postings that have needle_term."
        for span in self.find_spans(run, needle_term):
            for posting in self.read_span(span):
                if posting[0] == needle_term:
                    yield posting
                # Once we reach an alphabetically later term, we're done:
                elif posting[0] > needle_term:
                    break

    def read_span(self, span):
        with gzip.open(span, 'rt') as f:
            for posting in read_tuples(f):
                if posting[1] in self.documents:
                    yield posting

    def find_spans(self, run, term):
        "Yield the spans within the run that may have the term."
        lo = ''
        for span_name, hi in self.muster_spans(run):
            if term < lo:
                break
            if term <= hi:
                yield join(run, span_name)
            lo = hi

    def muster_spans(self, run):
        with open(join(run, 'span-map')) as f:
            for ii, (hi,) in enumerate(read_tuples(f)):
                yield '%s.gz' % ii, hi

span_size = 3 if debug else 4096

# 2**20 is chosen as the maximum run size because that uses
# typically about a quarter gig, which is a reasonable size these
# days.
run_size = 10 if debug else 2**20

# TODO: review how contextmanager works. do I need try-finally?
# XXX this grouping rather sucks. Move more logic into the class?
@contextmanager
def open_for_update(path):
    try:
        write_lock = open(join(path, 'write_lock'), 'xb')
    except FileExistsError:
        raise Exception("The index at %s seems to be busy on another update."
                        % path)
    index = WritableIndex(path)
    try:
        yield index
        index.commit(write_lock)
    except: # Or roll it back.
        os.remove(write_lock.name)
        raise
    else:
        index.clean_up()

class WritableIndex(Index):
    def __init__(self, path):
        Index.__init__(self, path)
        self.trash = []         # Directories to delete

    def update(self, corpus_path, term_transform):
        updates = self.muster_updates(corpus_path)
        postings = self.tokenize_documents(updates, term_transform)
        # Write all the new postings, first into runs, then merged.
        for run_postings in chunked(postings, run_size):
            self.write_new_run(sorted(run_postings))
        self.merge_some_runs()

    def commit(self, write_lock):
        "Commit to the updated catalog."
        self.write_catalog(write_lock)
        os.rename(write_lock.name, self.catalog_path)
        write_lock.close()

    def clean_up(self):
        for run in self.trash:
            shutil.rmtree(run)

    def write_catalog(self, catalog_file):
        pickle.dump({'runs': {int(basename(run)): size
                              for run, size in self.runs.items()},
                     'documents': self.documents,
                     'next_id': next(self.next_doc_ids)},
                    catalog_file)

    def muster_updates(self, corpus_path):
        """Yield the documents under corpus_path that have changed since the
        last reindexing, updating the catalog for them as we go."""
        paths = list(muster_files(self.to_relpath(corpus_path)))
        #print('mustard from', corpus_path, self.to_relpath(corpus_path), ':', paths)

        # Note deleted documents.
        for catalogued in set(self.doc_ids.keys()) - set(paths):
            if is_under(corpus_path, catalogued):
                del self.documents[self.doc_ids[catalogued]]
                del self.doc_ids[catalogued]
                if debug: print('missing', basename(catalogued))

        # Note and yield new and changed documents.
        for path in paths:
            metadata = get_metadata(path)
            old_doc_id = self.doc_ids.get(path)
            if old_doc_id is None or self.doc_metadata(old_doc_id) != metadata:
                if old_doc_id is not None:
                    del self.documents[old_doc_id]
                self.doc_ids[path] = doc_id = next(self.next_doc_ids)
                self.documents[doc_id] = (path, None, metadata)
                if debug: print('indexing', basename(path))
                yield path, doc_id

    def tokenize_documents(self, pairs, term_transform):
        for path, doc_id in pairs:
            term_counter = Counter(term_transform(tokenize_file(path)))
            path, _, metadata = self.documents[doc_id]
            self.documents[doc_id] = path, sum(term_counter.values()), metadata
            for term, count in term_counter.items():
                yield term, doc_id, str(count)

    def merge_some_runs(self):
        """Combine the smaller runs into one; but stop short of merging only a
        little data with a lot."""
        mergees, sizes = zip(*sorted(self.runs.items(),
                                     key=lambda item: item[1]))
        total, overflow = 0, None
        for ii, size in enumerate(sizes):
            if size <= total: overflow = ii
            total += size
        if overflow is not None:
            merger = mergees[:overflow+1]
            self.write_new_run(heapq.merge(*map(self.read_run, merger)))
            self.trash.extend(merger)
            for run in merger:
                del self.runs[run]

    def write_new_run(self, postings):
        # Pre: postings is sorted.
        run = self.new_run()
        os.mkdir(run)
        size = 0
        with open(join(run, 'span-map'), 'w') as map_file:
            for ii, span_postings in enumerate(chunked(postings, span_size)):
                span_postings = list(span_postings)
                with gzip.open(join(run, '%s.gz' % ii), 'wt') as span_file:
                    write_tuples(span_file, span_postings)
                write_tuple(map_file, (span_postings[-1][0],))
                size += len(span_postings)
        self.runs[run] = size

    def new_run(self):
        return join(self.my_path, str(next(self.next_runs)))


# Tokenizing

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
