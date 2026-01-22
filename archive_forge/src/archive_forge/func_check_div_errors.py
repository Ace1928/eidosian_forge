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
def check_div_errors(self, usecase_name, msg, flags=force_pyobj_flags, allow_complex=False):
    pyfunc = getattr(self.op, usecase_name)
    arg_types = [types.int32, types.uint32, types.float64]
    if allow_complex:
        arg_types.append(types.complex128)
    for tp in arg_types:
        cfunc = jit((tp, tp), **flags)(pyfunc)
        with self.assertRaises(ZeroDivisionError) as cm:
            cfunc(1, 0)
        if flags is not force_pyobj_flags:
            self.assertIn(msg, str(cm.exception))