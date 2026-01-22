from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def topickle(table, source=None, protocol=-1, write_header=True):
    """
    Write the table to a pickle file. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> etl.topickle(table1, 'example.p')
        >>> # look what it did
        ... table2 = etl.frompickle('example.p')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+
        | 'c' |   2 |
        +-----+-----+

    Note that if a file already exists at the given location, it will be
    overwritten.

    The pickle file format preserves type information, i.e., reading and writing
    is round-trippable for tables with non-string data values.

    """
    _writepickle(table, source=source, mode='wb', protocol=protocol, write_header=write_header)