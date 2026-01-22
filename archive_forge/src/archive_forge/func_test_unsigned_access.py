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
def test_unsigned_access(self):
    L = List.empty_list(int32)
    ui32_0 = types.uint32(0)
    ui32_1 = types.uint32(1)
    ui32_2 = types.uint32(2)
    L.append(types.uint32(10))
    L.append(types.uint32(11))
    L.append(types.uint32(12))
    self.assertEqual(len(L), 3)
    self.assertEqual(L[ui32_0], 10)
    self.assertEqual(L[ui32_1], 11)
    self.assertEqual(L[ui32_2], 12)
    L[ui32_0] = 123
    L[ui32_1] = 456
    L[ui32_2] = 789
    self.assertEqual(L[ui32_0], 123)
    self.assertEqual(L[ui32_1], 456)
    self.assertEqual(L[ui32_2], 789)
    ui32_123 = types.uint32(123)
    ui32_456 = types.uint32(456)
    ui32_789 = types.uint32(789)
    self.assertEqual(L.index(ui32_123), 0)
    self.assertEqual(L.index(ui32_456), 1)
    self.assertEqual(L.index(ui32_789), 2)
    L.__delitem__(ui32_2)
    del L[ui32_1]
    self.assertEqual(len(L), 1)
    self.assertEqual(L[ui32_0], 123)
    L.append(2)
    L.append(3)
    L.append(4)
    self.assertEqual(len(L), 4)
    self.assertEqual(L.pop(), 4)
    self.assertEqual(L.pop(ui32_2), 3)
    self.assertEqual(L.pop(ui32_1), 2)
    self.assertEqual(L.pop(ui32_0), 123)