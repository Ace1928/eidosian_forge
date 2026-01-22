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
def test_unboxer_basic(self):
    magic_token = 51966
    magic_offset = 123

    @intrinsic
    def my_intrinsic(typingctx, val):

        def impl(context, builder, sig, args):
            [val] = args
            return builder.add(val, val.type(magic_offset))
        sig = signature(val, val)
        return (sig, impl)

    @unbox(self.DynTypeType)
    def unboxer(typ, obj, c):

        def bridge(x):
            return my_intrinsic(x)
        args = [c.context.get_constant(types.intp, magic_token)]
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
    out = passthru(self.dyn_type)
    self.assertEqual(out, magic_token + magic_offset)