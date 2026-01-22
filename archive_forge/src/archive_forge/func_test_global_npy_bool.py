import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_npy_bool(self):
    pyfunc = global_npy_bool
    jitfunc = njit(pyfunc)
    self.assertEqual(pyfunc(), jitfunc())