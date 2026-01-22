from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def software_version(self):
    """Software instance and version that wrote this file"""
    return self.reader.software_version()