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
def test_invert(self):

    def control_signed(a):
        tp = self.get_numpy_signed_upcast(a)
        return tp(~a)

    def control_unsigned(a):
        tp = self.get_numpy_unsigned_upcast(a)
        return tp(~a)
    samples = self.int_samples
    pyfunc = self.op.bitwise_not_usecase
    self.run_unary(pyfunc, control_signed, samples, self.signed_types)
    self.run_unary(pyfunc, control_unsigned, samples, self.unsigned_types)