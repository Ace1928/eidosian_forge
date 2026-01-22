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
def test_catch_global_typed_list(self):
    from numba.tests.typedlist_usecases import catch_global
    expected_message = "The use of a ListType[int32] type, assigned to variable 'global_typed_list' in globals, is not supported as globals are considered compile-time constants and there is no known way to compile a ListType[int32] type as a constant."
    with self.assertRaises(TypingError) as raises:
        njit(catch_global)()
    self.assertIn(expected_message, str(raises.exception))
    self.disable_leak_check()