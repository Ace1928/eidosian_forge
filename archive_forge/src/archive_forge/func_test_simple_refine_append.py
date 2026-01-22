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
def test_simple_refine_append(self):

    @njit
    def foo():
        l = List()
        l.append(1)
        return l
    expected = foo.py_func()
    got = foo()
    self.assertEqual(expected, got)
    self.assertEqual(list(got), [1])
    self.assertEqual(typeof(got).item_type, typeof(1))