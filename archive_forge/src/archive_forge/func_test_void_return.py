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
def test_void_return(self):
    """
        Verify that returning a None from codegen function is handled
        automatically for void functions, otherwise raise exception.
        """

    @intrinsic
    def void_func(typingctx, a):
        sig = types.void(types.int32)

        def codegen(context, builder, signature, args):
            pass
        return (sig, codegen)

    @intrinsic
    def non_void_func(typingctx, a):
        sig = types.int32(types.int32)

        def codegen(context, builder, signature, args):
            pass
        return (sig, codegen)

    @jit(nopython=True)
    def call_void_func():
        void_func(1)
        return 0

    @jit(nopython=True)
    def call_non_void_func():
        non_void_func(1)
        return 0
    self.assertEqual(call_void_func(), 0)
    with self.assertRaises(LoweringError) as e:
        call_non_void_func()
    self.assertIn('non-void function returns None', e.exception.msg)