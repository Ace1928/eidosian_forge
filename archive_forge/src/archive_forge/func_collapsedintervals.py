from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def collapsedintervals(table, start='start', stop='stop', key=None):
    """
    Utility function to collapse intervals in a table. 
    
    If no facet `key` is given, returns an iterator over `(start, stop)` tuples.
    
    If facet `key` is given, returns an iterator over `(key, start, stop)`
    tuples.
    
    """
    if key is None:
        table = sort(table, key=start)
        for iv in _collapse(values(table, (start, stop))):
            yield iv
    else:
        table = sort(table, key=(key, start))
        for k, g in rowgroupby(table, key=key, value=(start, stop)):
            for iv in _collapse(g):
                yield ((k,) + iv)