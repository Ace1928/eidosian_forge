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
def test_setitem_slice_value_error(self):
    self.disable_leak_check()
    tl = List.empty_list(int32)
    for i in range(10, 20):
        tl.append(i)
    assignment = List.empty_list(int32)
    for i in range(1, 4):
        assignment.append(i)
    with self.assertRaises(ValueError) as raises:
        tl[8:3:-1] = assignment
    self.assertIn('length mismatch for extended slice and sequence', str(raises.exception))