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
def test_mod_complex(self, flags=force_pyobj_flags):
    pyfunc = self.op.mod_usecase
    cres = jit((types.complex64, types.complex64), **flags)(pyfunc)
    with self.assertRaises(TypeError) as raises:
        cres(4j, 2j)
    if utils.PYVERSION in ((3, 9),):
        msg = "can't mod complex numbers"
    elif utils.PYVERSION in ((3, 10), (3, 11), (3, 12)):
        msg = 'unsupported operand type(s) for %'
    else:
        raise NotImplementedError(utils.PYVERSION)
    self.assertIn(msg, str(raises.exception))