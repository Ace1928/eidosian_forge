from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def facetintervalrecordlookupone(table, key, start, stop, include_stop=False, strict=True):
    """
    As :func:`petl.transform.intervals.facetintervallookupone` but return
    records.

    """
    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop, strict=strict)
    return out