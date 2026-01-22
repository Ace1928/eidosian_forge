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
def test_exception_on_kwargs(self):

    @njit
    def foo():
        l = List(iterable=(0, 1, 2))
        return l
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('List() takes no keyword arguments', str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        List(iterable=(0, 1, 2))
    self.assertIn('List() takes no keyword arguments', str(raises.exception))