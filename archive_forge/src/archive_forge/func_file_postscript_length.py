from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def file_postscript_length(self):
    """The number of bytes in the file postscript"""
    return self.reader.file_postscript_length()