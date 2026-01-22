from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def biselect(table, *args, **kwargs):
    """Return two tables, the first containing selected rows, the second
    containing remaining rows. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 4, 9.3],
        ...           ['a', 2, 88.2],
        ...           ['b', 1, 23.3],
        ...           ['c', 8, 42.0],
        ...           ['d', 7, 100.9],
        ...           ['c', 2]]
        >>> table2, table3 = etl.biselect(table1, lambda rec: rec.foo == 'a')
        >>> table2
        +-----+-----+------+
        | foo | bar | baz  |
        +=====+=====+======+
        | 'a' |   4 |  9.3 |
        +-----+-----+------+
        | 'a' |   2 | 88.2 |
        +-----+-----+------+
        >>> table3
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'b' |   1 |  23.3 |
        +-----+-----+-------+
        | 'c' |   8 |  42.0 |
        +-----+-----+-------+
        | 'd' |   7 | 100.9 |
        +-----+-----+-------+
        | 'c' |   2 |       |
        +-----+-----+-------+

    .. versionadded:: 1.1.0

    """
    kwargs['complement'] = False
    t1 = select(table, *args, **kwargs)
    kwargs['complement'] = True
    t2 = select(table, *args, **kwargs)
    return (t1, t2)