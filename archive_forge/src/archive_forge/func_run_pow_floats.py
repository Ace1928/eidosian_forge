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
def run_pow_floats(self, pyfunc, flags=force_pyobj_flags):
    x_operands = [-222.222, -111.111, 111.111, 222.222]
    y_operands = [-2, -1, 0, 1, 2]
    types_list = [(types.float32, types.float32), (types.float64, types.float64)]
    self.run_test_floats(pyfunc, x_operands, y_operands, types_list, flags=flags)
    x_operands = [0.0]
    y_operands = [0, 1, 2]
    types_list = [(types.float32, types.float32), (types.float64, types.float64)]
    self.run_test_floats(pyfunc, x_operands, y_operands, types_list, flags=flags)