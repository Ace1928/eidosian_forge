from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervaljoin(left, right, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False, lprefix=None, rprefix=None):
    """
    Join two tables by overlapping intervals. E.g.::

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
        >>> table1 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop')
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

        >>> # include stop coordinate in intervals
        ... table2 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop',
        ...                           include_stop=True)
        >>> table2.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    An additional key comparison can be made, e.g.::
    
        >>> import petl as etl
        >>> left = (('fruit', 'begin', 'end'),
        ...         ('apple', 1, 2),
        ...         ('apple', 2, 4),
        ...         ('apple', 2, 5),
        ...         ('orange', 2, 5),
        ...         ('orange', 9, 14),
        ...         ('orange', 19, 140),
        ...         ('apple', 1, 1))
        >>> right = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> table3 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end', lkey='fruit',
        ...                           rstart='start', rstop='stop', rkey='type')
        >>> table3.lookall()
        +----------+-------+-----+----------+-------+------+-------+
        | fruit    | begin | end | type     | start | stop | value |
        +==========+=======+=====+==========+=======+======+=======+
        | 'apple'  |     1 |   2 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'orange' |     2 |   5 | 'orange' |     4 |    9 | 'baz' |
        +----------+-------+-----+----------+-------+------+-------+

    """
    assert (lkey is None) == (rkey is None), 'facet key field must be provided for both or neither table'
    return IntervalJoinView(left, right, lstart=lstart, lstop=lstop, rstart=rstart, rstop=rstop, lkey=lkey, rkey=rkey, include_stop=include_stop, lprefix=lprefix, rprefix=rprefix)