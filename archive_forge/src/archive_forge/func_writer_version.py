from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def writer_version(self):
    """Version of the writer"""
    return self.reader.writer_version()