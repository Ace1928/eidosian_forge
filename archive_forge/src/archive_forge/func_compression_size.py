from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def compression_size(self):
    """Number of bytes to buffer for the compression codec in the file"""
    return self.reader.compression_size()