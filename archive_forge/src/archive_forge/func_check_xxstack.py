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
def check_xxstack(self, pyfunc, cfunc):
    """
        3d and 0d tests for hstack(), vstack(), dstack().
        """

    def generate_starargs():
        yield ()
    self.check_3d(pyfunc, cfunc, generate_starargs)
    a = np.array(42)
    b = np.array(-5j)
    c = np.array(True)
    self.check_stack(pyfunc, cfunc, (a, b, a))