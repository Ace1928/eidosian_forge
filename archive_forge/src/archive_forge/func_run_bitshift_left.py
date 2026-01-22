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
def run_bitshift_left(self, pyfunc, flags=force_pyobj_flags):
    x_operands = [0, 1]
    y_operands = [0, 1, 2, 4, 8, 16, 31]
    types_list = [(types.uint32, types.uint32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = [0, 1]
    y_operands = [0, 1, 2, 4, 8, 16, 32, 63]
    types_list = [(types.uint64, types.uint64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = [0, -1]
    y_operands = [0, 1, 2, 4, 8, 16, 31]
    types_list = [(types.int32, types.int32)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = [0, -1]
    y_operands = [0, 1, 2, 4, 8, 16, 32, 63]
    types_list = [(types.int64, types.int64)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)