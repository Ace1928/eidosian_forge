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
def test_sort_no_args(self):

    def udt(lst):
        lst.sort()
        return lst
    for nelem in [13, 29, 127]:
        my_lists = self.make_both(np.random.randint(0, nelem, nelem))
        self.assertEqual(list(udt(my_lists['nb'])), udt(my_lists['py']))