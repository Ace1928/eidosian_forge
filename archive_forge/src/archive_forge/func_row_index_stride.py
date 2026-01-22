from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def row_index_stride(self):
    """Number of rows per an entry in the row index or 0
        if there is no row index"""
    return self.reader.row_index_stride()