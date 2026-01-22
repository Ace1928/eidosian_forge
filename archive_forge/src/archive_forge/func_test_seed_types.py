from numba import float32, njit
import unittest
def test_seed_types(self):
    cfunc = njit((), locals={'x': float32})(foo)
    self.assertEqual(cfunc.nopython_signatures[0].return_type, float32)