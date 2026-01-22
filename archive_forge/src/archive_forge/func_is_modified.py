from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
@property
def is_modified(self):
    return self.action == FILE_ACTION_MODIFIED