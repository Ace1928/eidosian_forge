from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def stringpatterns(table, field):
    """
    Profile string patterns in the given field, returning a table of patterns,
    counts and frequencies. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['Mr. Foo', '123-1254'],
        ...          ['Mrs. Bar', '234-1123'],
        ...          ['Mr. Spo', '123-1254'],
        ...          [u'Mr. Baz', u'321 1434'],
        ...          [u'Mrs. Baz', u'321 1434'],
        ...          ['Mr. Quux', '123-1254-XX']]
        >>> etl.stringpatterns(table, 'foo')
        +------------+-------+---------------------+
        | pattern    | count | frequency           |
        +============+=======+=====================+
        | 'Aa. Aaa'  |     3 |                 0.5 |
        +------------+-------+---------------------+
        | 'Aaa. Aaa' |     2 |  0.3333333333333333 |
        +------------+-------+---------------------+
        | 'Aa. Aaaa' |     1 | 0.16666666666666666 |
        +------------+-------+---------------------+

        >>> etl.stringpatterns(table, 'bar')
        +---------------+-------+---------------------+
        | pattern       | count | frequency           |
        +===============+=======+=====================+
        | '999-9999'    |     3 |                 0.5 |
        +---------------+-------+---------------------+
        | '999 9999'    |     2 |  0.3333333333333333 |
        +---------------+-------+---------------------+
        | '999-9999-AA' |     1 | 0.16666666666666666 |
        +---------------+-------+---------------------+

    """
    counter = stringpatterncounter(table, field)
    output = [('pattern', 'count', 'frequency')]
    counter = counter.most_common()
    total = sum((c[1] for c in counter))
    cnts = [(c[0], c[1], float(c[1]) / total) for c in counter]
    output.extend(cnts)
    return wrap(output)