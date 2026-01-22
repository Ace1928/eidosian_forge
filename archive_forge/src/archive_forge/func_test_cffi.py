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
@skip_unless_cffi
def test_cffi(self):
    from numba.tests import cffi_usecases
    ffi, lib = cffi_usecases.load_inline_module()
    f = cfunc(square_sig)(square_usecase)
    res = lib._numba_test_funcptr(f.cffi)
    self.assertPreciseEqual(res, 2.25)