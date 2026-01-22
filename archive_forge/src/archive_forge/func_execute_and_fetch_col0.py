import os
from . import BioSeq
from . import Loader
from . import DBUtils
def execute_and_fetch_col0(self, sql, args=None):
    """Return a list of values from the first column in the row."""
    out = super().execute_and_fetch_col0(sql, args)
    return [self._bytearray_to_str(column) for column in out]