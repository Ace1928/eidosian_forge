from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def convertnumbers(table, strict=False, **kwargs):
    """
    Convenience function to convert all field values to numbers where
    possible. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz', 'quux'],
        ...           ['1', '3.0', '9+3j', 'aaa'],
        ...           ['2', '1.3', '7+2j', None]]
        >>> table2 = etl.convertnumbers(table1)
        >>> table2
        +-----+-----+--------+-------+
        | foo | bar | baz    | quux  |
        +=====+=====+========+=======+
        |   1 | 3.0 | (9+3j) | 'aaa' |
        +-----+-----+--------+-------+
        |   2 | 1.3 | (7+2j) | None  |
        +-----+-----+--------+-------+

    """
    return convertall(table, numparser(strict), **kwargs)