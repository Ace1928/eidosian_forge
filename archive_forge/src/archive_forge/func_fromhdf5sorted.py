from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def fromhdf5sorted(source, where=None, name=None, sortby=None, checkCSI=False, start=None, stop=None, step=None):
    """
    Provides access to an HDF5 table, sorted by an indexed column, e.g.::

        >>> import petl as etl
        >>>
        >>> # set up a new hdf5 table to demonstrate with
        >>> class FooBar(tables.IsDescription): # doctest: +SKIP
        ...     foo = tables.Int32Col(pos=0) # doctest: +SKIP
        ...     bar = tables.StringCol(6, pos=2) # doctest: +SKIP
        >>>
        >>> def setup_hdf5_index():
        ...     import tables
        ...     h5file = tables.open_file('example.h5', mode='w',
        ...                               title='Example file')
        ...     h5file.create_group('/', 'testgroup', 'Test Group')
        ...     h5table = h5file.create_table('/testgroup', 'testtable', FooBar,
        ...                                   'Test Table')
        ...     # load some data into the table
        ...     table1 = (('foo', 'bar'),
        ...               (1, b'asdfgh'),
        ...               (2, b'qwerty'),
        ...               (3, b'zxcvbn'))
        ...     for row in table1[1:]:
        ...         for i, f in enumerate(table1[0]):
        ...             h5table.row[f] = row[i]
        ...         h5table.row.append()
        ...     h5table.cols.foo.create_csindex()  # CS index is required
        ...     h5file.flush()
        ...     h5file.close()
        >>>
        >>> setup_hdf5_index() # doctest: +SKIP
        >>>
        ... # access the data, sorted by the indexed column
        ... table2 = etl.fromhdf5sorted('example.h5', '/testgroup', 'testtable', sortby='foo') # doctest: +SKIP
        >>> table2 # doctest: +SKIP
        +-----+-----------+
        | foo | bar       |
        +=====+===========+
        |   1 | b'zxcvbn' |
        +-----+-----------+
        |   2 | b'qwerty' |
        +-----+-----------+
        |   3 | b'asdfgh' |
        +-----+-----------+

    """
    assert sortby is not None, 'no column specified to sort by'
    return HDF5SortedView(source, where=where, name=name, sortby=sortby, checkCSI=checkCSI, start=start, stop=stop, step=step)