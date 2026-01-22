from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
def iterduplicates(source, key):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        if key is None:
            return
        hdr = []
    yield tuple(hdr)
    if key is None:
        indices = range(len(hdr))
    else:
        indices = asindices(hdr, key)
    getkey = operator.itemgetter(*indices)
    previous = None
    previous_yielded = False
    for row in it:
        if previous is None:
            previous = row
        else:
            kprev = getkey(previous)
            kcurr = getkey(row)
            if kprev == kcurr:
                if not previous_yielded:
                    yield tuple(previous)
                    previous_yielded = True
                yield tuple(row)
            else:
                previous_yielded = False
            previous = row