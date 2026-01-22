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
def test_mod_complex_npm(self):
    pyfunc = self.op.mod_usecase
    with self.assertTypingError():
        njit((types.complex64, types.complex64))(pyfunc)