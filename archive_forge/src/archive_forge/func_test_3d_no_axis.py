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
def test_3d_no_axis(self):
    """
        stack(3d arrays)
        """
    pyfunc = np_stack1
    cfunc = nrtjit(pyfunc)

    def generate_starargs():
        yield ()
    self.check_3d(pyfunc, cfunc, generate_starargs)
    self.check_runtime_errors(cfunc, generate_starargs)