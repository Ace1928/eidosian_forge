from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def tupletree(table, start='start', stop='stop', value=None):
    """
    Construct an interval tree for the given table, where each node in the tree
    is a row of the table.

    """
    import intervaltree
    tree = intervaltree.IntervalTree()
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    assert start in flds, 'start field not recognised'
    assert stop in flds, 'stop field not recognised'
    getstart = itemgetter(flds.index(start))
    getstop = itemgetter(flds.index(stop))
    if value is None:
        getvalue = tuple
    else:
        valueindices = asindices(hdr, value)
        assert len(valueindices) > 0, 'invalid value field specification'
        getvalue = itemgetter(*valueindices)
    for row in it:
        tree.addi(getstart(row), getstop(row), getvalue(row))
    return tree