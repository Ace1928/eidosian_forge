import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def iterparse(source, events=None, parser=None):
    """Incrementally parse XML document into ElementTree.

    This class also reports what's going on to the user based on the
    *events* it is initialized with.  The supported events are the strings
    "start", "end", "start-ns" and "end-ns" (the "ns" events are used to get
    detailed namespace information).  If *events* is omitted, only
    "end" events are reported.

    *source* is a filename or file object containing XML data, *events* is
    a list of events to report back, *parser* is an optional parser instance.

    Returns an iterator providing (event, elem) pairs.

    """
    pullparser = XMLPullParser(events=events, _parser=parser)
    if not hasattr(source, 'read'):
        source = open(source, 'rb')
        close_source = True
    else:
        close_source = False

    def iterator(source):
        try:
            while True:
                yield from pullparser.read_events()
                data = source.read(16 * 1024)
                if not data:
                    break
                pullparser.feed(data)
            root = pullparser._close_and_return_root()
            yield from pullparser.read_events()
            it = wr()
            if it is not None:
                it.root = root
        finally:
            if close_source:
                source.close()

    class IterParseIterator(collections.abc.Iterator):
        __next__ = iterator(source).__next__

        def __del__(self):
            if close_source:
                source.close()
    it = IterParseIterator()
    it.root = None
    wr = weakref.ref(it)
    return it