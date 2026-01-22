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
def test_sort_all_args(self):

    def udt(lst, key, reverse):
        lst.sort(key=key, reverse=reverse)
        return lst
    possible_keys = [lambda x: -x, lambda x: 1 / (1 + x), lambda x: (x, -x), lambda x: x]
    possible_reverse = [True, False]
    for key, reverse in product(possible_keys, possible_reverse):
        my_lists = self.make_both(np.random.randint(0, 100, 23))
        msg = 'case for key={} reverse={}'.format(key, reverse)
        self.assertEqual(list(udt(my_lists['nb'], key=key, reverse=reverse)), udt(my_lists['py'], key=key, reverse=reverse), msg=msg)