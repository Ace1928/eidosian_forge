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
def run_logical(self, pyfunc, flags=force_pyobj_flags):
    x_operands = list(range(0, 8)) + [2 ** 32 - 1]
    y_operands = list(range(0, 8)) + [2 ** 32 - 1]
    types_list = [(types.uint32, types.uint32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(0, 8)) + [2 ** 64 - 1]
    y_operands = list(range(0, 8)) + [2 ** 64 - 1]
    types_list = [(types.uint64, types.uint64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(-4, 4)) + [-2 ** 31, 2 ** 31 - 1]
    y_operands = list(range(-4, 4)) + [-2 ** 31, 2 ** 31 - 1]
    types_list = [(types.int32, types.int32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = list(range(-4, 4)) + [-2 ** 63, 2 ** 63 - 1]
    y_operands = list(range(-4, 4)) + [-2 ** 63, 2 ** 63 - 1]
    types_list = [(types.int64, types.int64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)