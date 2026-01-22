import gc
from numba import jit, int32
import unittest
def test_bar_call_foo(self):
    global cfoo
    cfoo = jit((int32, int32), nopython=True)(foo)
    cbar = jit((int32, int32), nopython=True)(bar)
    self.assertEqual(cbar(1, 2), 1 + 2 + 2)