import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase
def gen_ir(self, func, args_tuple, fastmath=False):
    self.assertEqual(config.CPU_NAME, 'skylake-avx512')
    self.assertEqual(config.CPU_FEATURES, '')
    jitted = njit(args_tuple, fastmath=fastmath)(func)
    return jitted.inspect_llvm(args_tuple)