from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def tohdf5(table, source, where=None, name=None, create=False, drop=False, description=None, title='', filters=None, expectedrows=10000, chunkshape=None, byteorder=None, createparents=False, sample=1000):
    """
    Write to an HDF5 table. If `create` is `False`, assumes the table
    already exists, and attempts to truncate it before loading. If `create`
    is `True`, a new table will be created, and if `drop` is True,
    any existing table will be dropped first. If `description` is `None`,
    the description will be guessed. E.g.::

        >>> import petl as etl
        >>> table1 = (('foo', 'bar'),
        ...           (1, b'asdfgh'),
        ...           (2, b'qwerty'),
        ...           (3, b'zxcvbn'))
        >>> etl.tohdf5(table1, 'example.h5', '/testgroup', 'testtable',
        ...            drop=True, create=True, createparents=True) # doctest: +SKIP
        >>> etl.fromhdf5('example.h5', '/testgroup', 'testtable') # doctest: +SKIP
        +-----+-----------+
        | foo | bar       |
        +=====+===========+
        |   1 | b'asdfgh' |
        +-----+-----------+
        |   2 | b'qwerty' |
        +-----+-----------+
        |   3 | b'zxcvbn' |
        +-----+-----------+

    """
    import tables
    it = iter(table)
    if create:
        with _get_hdf5_file(source, mode='a') as h5file:
            if drop:
                try:
                    h5file.get_node(where, name)
                except tables.NoSuchNodeError:
                    pass
                else:
                    h5file.remove_node(where, name)
            if description is None:
                peek, it = iterpeek(it, sample)
                description = infer_dtype(peek)
            h5file.create_table(where, name, description, title=title, filters=filters, expectedrows=expectedrows, chunkshape=chunkshape, byteorder=byteorder, createparents=createparents)
    with _get_hdf5_table(source, where, name, mode='a') as h5table:
        h5table.truncate(0)
        _insert(it, h5table)