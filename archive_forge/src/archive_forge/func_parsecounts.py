from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def parsecounts(table, field, parsers=(('int', int), ('float', float))):
    """
    Count the number of `str` or `unicode` values that can be parsed as ints,
    floats or via custom parser functions. Return a table mapping parser names
    to the number of values successfully parsed and the number of errors. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['A', 'aaa', 2],
        ...          ['B', u'2', '3.4'],
        ...          [u'B', u'3', u'7.8', True],
        ...          ['D', '3.7', 9.0],
        ...          ['E', 42]]
        >>> etl.parsecounts(table, 'bar')
        +---------+-------+--------+
        | type    | count | errors |
        +=========+=======+========+
        | 'float' |     3 |      1 |
        +---------+-------+--------+
        | 'int'   |     2 |      2 |
        +---------+-------+--------+

    The `field` argument can be a field name or index (starting from zero).

    """
    return ParseCountsView(table, field, parsers=parsers)