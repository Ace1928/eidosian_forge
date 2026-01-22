import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def supports_python(self):
    """Returns True if the underlying gdb implementation has python support
           False otherwise"""
    return 'python' in self.list_features()