import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_array_pop_from_single_value_list(self):

    @njit
    def foo():
        l = List((np.zeros((1,)),))
        l.pop()
        return l
    expected, got = (foo.py_func(), foo())
    self.assertEqual(len(expected), 0)
    self.assertEqual(len(got), 0)