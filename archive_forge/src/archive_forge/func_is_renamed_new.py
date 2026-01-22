from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
@property
def is_renamed_new(self):
    return self.action == FILE_ACTION_RENAMED_NEW_NAME