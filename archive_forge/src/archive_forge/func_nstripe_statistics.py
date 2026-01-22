from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def nstripe_statistics(self):
    """Number of stripe statistics"""
    return self.reader.nstripe_statistics()