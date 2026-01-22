import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_issue_with_return_leak(self):
    """
        Dispatcher returns a new reference.
        It need to workaround it for now.
        """

    @nrtjit
    def inner(out):
        return out

    def pyfunc(x):
        return inner(x)
    cfunc = nrtjit(pyfunc)
    arr = np.arange(10)
    old_refct = sys.getrefcount(arr)
    self.assertEqual(old_refct, sys.getrefcount(pyfunc(arr)))
    self.assertEqual(old_refct, sys.getrefcount(cfunc(arr)))
    self.assertEqual(old_refct, sys.getrefcount(arr))