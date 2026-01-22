from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def extendheader(table, fields):
    """
    Extend header row in the given table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo'],
        ...           ['a', 1, True],
        ...           ['b', 2, False]]
        >>> table2 = etl.extendheader(table1, ['bar', 'baz'])
        >>> table2
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'a' |   1 | True  |
        +-----+-----+-------+
        | 'b' |   2 | False |
        +-----+-----+-------+

    See also :func:`petl.transform.headers.setheader`,
    :func:`petl.transform.headers.pushheader`.

    """
    return ExtendHeaderView(table, fields)