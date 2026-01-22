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
def test_object_mode(self):
    """
        When the function is compiled in object mode, a warning is
        printed out.
        """
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always', errors.NumbaWarning)
        cfunc = jit(f_sig, forceobj=True, nogil=True)(object_f)
    self.assertTrue(any((w.category is errors.NumbaWarning and "Code running in object mode won't allow parallel execution" in str(w.message) for w in wlist)), wlist)
    self.run_in_threads(cfunc, 2)