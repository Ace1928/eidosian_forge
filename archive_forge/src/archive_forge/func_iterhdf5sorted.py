from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def iterhdf5sorted(source, where, name, sortby, checkCSI, start, stop, step):
    with _get_hdf5_table(source, where, name) as h5tbl:
        hdr = tuple(h5tbl.colnames)
        yield hdr
        it = h5tbl.itersorted(sortby, checkCSI=checkCSI, start=start, stop=stop, step=step)
        for row in it:
            yield row[:]