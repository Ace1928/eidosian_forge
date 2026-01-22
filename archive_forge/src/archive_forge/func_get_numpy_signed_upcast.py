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
def get_numpy_signed_upcast(self, *vals):
    bitwidth = max((v.dtype.itemsize * 8 for v in vals))
    bitwidth = max(bitwidth, types.intp.bitwidth)
    return getattr(np, 'int%d' % bitwidth)