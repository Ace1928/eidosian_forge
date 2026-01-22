from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def facetintervalrecordlookup(table, key, start='start', stop='stop', include_stop=False):
    """
    As :func:`petl.transform.intervals.facetintervallookup` but return records.
    
    """
    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out