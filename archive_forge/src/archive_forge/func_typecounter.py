from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def typecounter(table, field):
    """
    Count the number of values found for each Python type.

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['A', 1, 2],
        ...          ['B', u'2', '3.4'],
        ...          [u'B', u'3', u'7.8', True],
        ...          ['D', u'xyz', 9.0],
        ...          ['E', 42]]
        >>> etl.typecounter(table, 'foo')
        Counter({'str': 5})
        >>> etl.typecounter(table, 'bar')
        Counter({'str': 3, 'int': 2})
        >>> etl.typecounter(table, 'baz')
        Counter({'str': 2, 'int': 1, 'float': 1, 'NoneType': 1})

    The `field` argument can be a field name or index (starting from zero).

    """
    counter = Counter()
    for v in values(table, field):
        try:
            counter[v.__class__.__name__] += 1
        except IndexError:
            pass
    return counter