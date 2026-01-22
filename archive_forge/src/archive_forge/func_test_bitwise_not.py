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
def test_bitwise_not(self, flags=force_pyobj_flags):
    pyfunc = self.op.bitwise_not_usecase_binary
    x_operands = list(range(0, 8)) + [2 ** 32 - 1]
    x_operands = [np.uint32(x) for x in x_operands]
    y_operands = [0]
    types_list = [(types.uint32, types.uint32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(-4, 4)) + [-2 ** 31, 2 ** 31 - 1]
    y_operands = [0]
    types_list = [(types.int32, types.int32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(0, 8)) + [2 ** 64 - 1]
    x_operands = [np.uint64(x) for x in x_operands]
    y_operands = [0]
    types_list = [(types.uint64, types.uint64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(-4, 4)) + [-2 ** 63, 2 ** 63 - 1]
    y_operands = [0]
    types_list = [(types.int64, types.int64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    values = [False, False, True, True]
    values = list(map(np.bool_, values))
    pyfunc = self.op.bitwise_not_usecase
    cfunc = jit((types.boolean,), **flags)(pyfunc)
    for val in values:
        self.assertPreciseEqual(pyfunc(val), cfunc(val))