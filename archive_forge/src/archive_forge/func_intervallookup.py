from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervallookup(table, start='start', stop='stop', value=None, include_stop=False):
    """
    Construct an interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop')
        >>> lkp.search(0, 1)
        []
        >>> lkp.search(1, 2)
        [(1, 4, 'foo')]
        >>> lkp.search(2, 4)
        [(1, 4, 'foo'), (3, 7, 'bar')]
        >>> lkp.search(2, 5)
        [(1, 4, 'foo'), (3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(9, 14)
        []
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        [(1, 4, 'foo')]
        >>> lkp.search(2)
        [(1, 4, 'foo')]
        >>> lkp.search(4)
        [(3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(5)
        [(3, 7, 'bar'), (4, 9, 'baz')]

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    Some examples using the `include_stop` and `value` keyword arguments::
    
        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop', include_stop=True,
        ...                          value='value')
        >>> lkp.search(0, 1)
        ['foo']
        >>> lkp.search(1, 2)
        ['foo']
        >>> lkp.search(2, 4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(2, 5)
        ['foo', 'bar', 'baz']
        >>> lkp.search(9, 14)
        ['baz']
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        ['foo']
        >>> lkp.search(2)
        ['foo']
        >>> lkp.search(4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(5)
        ['bar', 'baz']

    """
    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookup(tree, include_stop=include_stop)