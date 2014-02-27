* Use Path object to simplify path manipulation.
* Consistently use "doc_id" for document IDs (even though they happen
  to be pathnames)
* Be consistent about "segment", "chunk", and "index".  And "posting".
* Reduce the maximum segment size from 2**22 to 2**20.  It's using
  half a gig (Oh shit now 600MB, oh shit 700MB, 726MB) now that all
  the postings are unique, and the segments are now like 27 megabytes!
* Actually make incremental indexing work.
    * When adding new segments, use a sensible merge policy.  Maybe
      shorten the explanation of this in the chapter.
    * Keep track of enough metadata about indexed files to tell when
      they're outdated, or when new files have been added.  Maybe
      inside each segment?
    * Somehow delete old postings.  Maybe blacklist outdated docids
      when merging segments.
* Split up chunks into multiple subdirectories, like Git.
* Track deleted postings so the index doesn't grow without bound.
* Find indices on the path to root and update them automatically,
  perhaps niced in the background.
* Perhaps some kind of abstraction of different pieces: storage
  engine, term extraction engine, etc.  Term extraction is already
  pretty abstract, in the sense that everything else just treats its
  output as a stream of (term, docid) tuples.
* Deal with fsync and incomplete segments to enable concurrency and
  crashes.
* Maybe encode output in UTF-8 to see if that gets it to run in
  Python3?
* Don't be case-sensitive!
* Maybe make more things be methods of an index object?
* See if it runs in Jython 2.5.  And PyPy.