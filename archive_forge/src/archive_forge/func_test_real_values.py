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
def test_real_values(self):
    exponents = [1, 2, 3, 5, 17, 0, -1, -2, -3, 1118481, -1118482]
    vals = [1.5, 3.25, -1.25, np.float32(-2.0), float('inf'), float('nan')]
    self._check_pow(exponents, vals)