from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def facetintervallookupone(table, key, start='start', stop='stop', value=None, include_stop=False, strict=True):
    """
    Construct a faceted interval lookup for the given table, returning at most
    one result for each query.
    
    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    """
    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop, strict=strict)
    return out