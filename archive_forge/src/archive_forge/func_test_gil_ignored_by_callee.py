import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
def test_gil_ignored_by_callee(self):
    """
        When only the callee asks to release the GIL, it gets ignored.
        """
    compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

    @jit(f_sig, nopython=True)
    def caller(a, i):
        compiled_f(a, i)
    self.check_gil_held(caller)