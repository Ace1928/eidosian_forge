from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def frompickle(source=None):
    """
    Extract a table From data pickled in the given file. The rows in the
    table should have been pickled to the file one at a time. E.g.::

        >>> import petl as etl
        >>> import pickle
        >>> # set up a file to demonstrate with
        ... with open('example.p', 'wb') as f:
        ...     pickle.dump(['foo', 'bar'], f)
        ...     pickle.dump(['a', 1], f)
        ...     pickle.dump(['b', 2], f)
        ...     pickle.dump(['c', 2.5], f)
        ...
        >>> # now demonstrate the use of frompickle()
        ... table1 = etl.frompickle('example.p')
        >>> table1
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+
        | 'c' | 2.5 |
        +-----+-----+


    """
    source = read_source_from_arg(source)
    return PickleView(source)