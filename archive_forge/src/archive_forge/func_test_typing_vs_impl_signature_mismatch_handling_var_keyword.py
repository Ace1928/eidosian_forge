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
def test_typing_vs_impl_signature_mismatch_handling_var_keyword(self):
    """
        Tests that an overload which uses **kwargs (VAR_KEYWORD)
        """

    def gen_ol(impl, strict=True):

        def myoverload(a, kw=None):
            pass
        overload(myoverload, strict=strict)(impl)

        @jit(nopython=True)
        def foo(a, b):
            return myoverload(a, kw=11)
        return foo

    def ol1(a, **kws):

        def impl(a, kw=10):
            return a
        return impl
    gen_ol(ol1, False)(1, 2)
    with self.assertRaises(errors.TypingError) as e:
        gen_ol(ol1)(1, 2)
    msg = str(e.exception)
    self.assertIn('use of VAR_KEYWORD (e.g. **kwargs) is unsupported', msg)
    self.assertIn("offending argument name is '**kws'", msg)

    def ol2(a, kw=0):

        def impl(a, **kws):
            return a
        return impl
    with self.assertRaises(errors.TypingError) as e:
        gen_ol(ol2)(1, 2)
    msg = str(e.exception)
    self.assertIn('use of VAR_KEYWORD (e.g. **kwargs) is unsupported', msg)
    self.assertIn("offending argument name is '**kws'", msg)