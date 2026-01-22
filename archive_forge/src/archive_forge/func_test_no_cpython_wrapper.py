import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_no_cpython_wrapper(self):
    """
        Test overloading whose return value cannot be represented in CPython.
        """
    ok_cfunc = jit(nopython=True)(non_boxable_ok_usecase)
    n = 10
    got = ok_cfunc(n)
    expect = non_boxable_ok_usecase(n)
    np.testing.assert_equal(expect, got)
    bad_cfunc = jit(nopython=True)(non_boxable_bad_usecase)
    with self.assertRaises(TypeError) as raises:
        bad_cfunc()
    errmsg = str(raises.exception)
    expectmsg = 'cannot convert native Module'
    self.assertIn(expectmsg, errmsg)