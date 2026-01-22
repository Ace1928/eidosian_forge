from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervalleftjoin(left, right, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False, missing=None, lprefix=None, rprefix=None):
    """
    Like :func:`petl.transform.intervals.intervaljoin` but rows from the left 
    table without a match in the right table are also included. E.g.::

        >>> import petl as etl
        >>> left = [['begin', 'end', 'quux'],
        ...         [1, 2, 'a'],
        ...         [2, 4, 'b'],
        ...         [2, 5, 'c'],
        ...         [9, 14, 'd'],
        ...         [1, 1, 'e'],
        ...         [10, 10, 'f']]
        >>> right = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> table1 = etl.intervalleftjoin(left, right,
        ...                               lstart='begin', lstop='end',
        ...                               rstart='start', rstop='stop')
        >>> table1.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |    10 |  10 | 'f'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    assert (lkey is None) == (rkey is None), 'facet key field must be provided for both or neither table'
    return IntervalLeftJoinView(left, right, lstart=lstart, lstop=lstop, rstart=rstart, rstop=rstop, lkey=lkey, rkey=rkey, include_stop=include_stop, missing=missing, lprefix=lprefix, rprefix=rprefix)