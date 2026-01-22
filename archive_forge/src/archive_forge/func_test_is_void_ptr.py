import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def test_is_void_ptr(self):
    cfunc_void = jit((types.voidptr, types.voidptr), nopython=True)(self.op.is_usecase)

    @jit(nopython=True)
    def cfunc(x, y):
        return cfunc_void(x, y)
    self.assertTrue(cfunc(1, 1))
    self.assertFalse(cfunc(1, 2))