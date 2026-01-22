from numba import njit
from functools import reduce
import unittest
def test_basic_map_closure(self):

    def impl():
        return [y for y in map(lambda x: x + 10, range(10))]
    cfunc = njit(impl)
    self.assertEqual(impl(), cfunc())