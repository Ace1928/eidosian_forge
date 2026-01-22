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
def test_list_as_value_in_dict(self):

    @njit
    def foo():
        d = Dict.empty(int32, List.empty_list(int32))
        l = List.empty_list(int32)
        l.append(0)
        d[0] = l
        return get_refcount(l)
    c = foo()
    if config.LLVM_REFPRUNE_PASS:
        self.assertEqual(1, c)
    else:
        self.assertEqual(2, c)