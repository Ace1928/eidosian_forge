from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervallookupone(table, start='start', stop='stop', value=None, include_stop=False, strict=True):
    """
    Construct an interval lookup for the given table, returning at most one
    result for each query. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookupone(table, 'start', 'stop', strict=False)
        >>> lkp.search(0, 1)
        >>> lkp.search(1, 2)
        (1, 4, 'foo')
        >>> lkp.search(2, 4)
        (1, 4, 'foo')
        >>> lkp.search(2, 5)
        (1, 4, 'foo')
        >>> lkp.search(9, 14)
        >>> lkp.search(19, 140)
        >>> lkp.search(0)
        >>> lkp.search(1)
        (1, 4, 'foo')
        >>> lkp.search(2)
        (1, 4, 'foo')
        >>> lkp.search(4)
        (3, 7, 'bar')
        >>> lkp.search(5)
        (3, 7, 'bar')

    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookupOne(tree, strict=strict, include_stop=include_stop)