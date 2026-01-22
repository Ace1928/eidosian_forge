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
def test_array_as_item_in_list(self):
    nested_type = types.Array(types.float64, 1, 'C')

    @njit
    def foo():
        l = List.empty_list(nested_type)
        a = np.zeros((1,))
        l.append(a)
        return l
    expected = foo.py_func()
    got = foo()
    self.assertTrue(np.all(expected[0] == got[0]))