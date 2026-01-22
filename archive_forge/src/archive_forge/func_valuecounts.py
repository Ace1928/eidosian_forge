from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def valuecounts(table, *field, **kwargs):
    """
    Find distinct values for the given field and count the number and relative
    frequency of occurrences. Returns a table mapping values to counts, with
    most common values first. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['a', True, 0.12],
        ...          ['a', True, 0.17],
        ...          ['b', False, 0.34],
        ...          ['b', False, 0.44],
        ...          ['b']]
        >>> etl.valuecounts(table, 'foo')
        +-----+-------+-----------+
        | foo | count | frequency |
        +=====+=======+===========+
        | 'b' |     3 |       0.6 |
        +-----+-------+-----------+
        | 'a' |     2 |       0.4 |
        +-----+-------+-----------+

        >>> etl.valuecounts(table, 'foo', 'bar')
        +-----+-------+-------+-----------+
        | foo | bar   | count | frequency |
        +=====+=======+=======+===========+
        | 'a' | True  |     2 |       0.4 |
        +-----+-------+-------+-----------+
        | 'b' | False |     2 |       0.4 |
        +-----+-------+-------+-----------+
        | 'b' | None  |     1 |       0.2 |
        +-----+-------+-------+-----------+

    If rows are short, the value of the keyword argument `missing` is counted.

    Multiple fields can be given as positional arguments. If multiple fields are
    given, these are treated as a compound key.

    """
    return ValueCountsView(table, field, **kwargs)