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
def test_unboxer_raise(self):

    @unbox(self.DynTypeType)
    def unboxer(typ, obj, c):

        def bridge(x):
            if x > 0:
                raise ValueError('cannot be x > 0')
            return x
        args = [c.context.get_constant(types.intp, 1)]
        sig = signature(types.voidptr, types.intp)
        is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
        return NativeValue(res, is_error=is_error)

    @box(self.DynTypeType)
    def boxer(typ, val, c):
        res = c.builder.ptrtoint(val, cgutils.intp_t)
        return c.pyapi.long_from_ssize_t(res)

    @njit
    def passthru(x):
        return x
    with self.assertRaises(ValueError) as raises:
        passthru(self.dyn_type)
    self.assertIn('cannot be x > 0', str(raises.exception))