import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def test_unsupport_bitsize(self):
    ffi = self.get_ffi()
    with self.assertRaises(ValueError) as raises:
        cffi_support.map_type(ffi.typeof('error'), use_record_dtype=True)
    self.assertEqual("field 'bits' has bitshift, this is not supported", str(raises.exception))