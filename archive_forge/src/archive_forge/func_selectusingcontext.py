from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def selectusingcontext(table, query):
    """
    Select rows based on data in the current row and/or previous and
    next row. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 1],
        ...           ['B', 4],
        ...           ['C', 5],
        ...           ['D', 9]]
        >>> def query(prv, cur, nxt):
        ...     return ((prv is not None and (cur.bar - prv.bar) < 2)
        ...             or (nxt is not None and (nxt.bar - cur.bar) < 2))
        ...
        >>> table2 = etl.selectusingcontext(table1, query)
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'B' |   4 |
        +-----+-----+
        | 'C' |   5 |
        +-----+-----+

    The `query` function should accept three rows and return a boolean value.

    """
    return SelectUsingContextView(table, query)