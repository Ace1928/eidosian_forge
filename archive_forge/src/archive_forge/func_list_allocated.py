import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_allocated(self):
    return self.tc.numba_list_allocated(self.lp)