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
def run_test_ints(self, pyfunc, x_operands, y_operands, types_list, flags=force_pyobj_flags):
    for arg_types in types_list:
        cfunc = jit(arg_types, **flags)(pyfunc)
        for x, y in itertools.product(x_operands, y_operands):
            x_got = copy.copy(x)
            x_expected = copy.copy(x)
            got = cfunc(x_got, y)
            expected = pyfunc(x_expected, y)
            self.assertPreciseEqual(got, expected, msg='mismatch for (%r, %r) with types %s: %r != %r' % (x, y, arg_types, got, expected))
            self.assertPreciseEqual(x_got, x_expected, msg='mismatch for (%r, %r) with types %s: %r != %r' % (x, y, arg_types, x_got, x_expected))