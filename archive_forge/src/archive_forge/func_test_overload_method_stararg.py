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
def test_overload_method_stararg(self):

    @overload_method(MyDummyType, 'method_stararg')
    def _ov_method_stararg(obj, val, val2, *args):

        def get(obj, val, val2, *args):
            return (val, val2, args)
        return get

    @njit
    def foo(obj, *args):
        return obj.method_stararg(*args)
    obj = MyDummy()
    self.assertEqual(foo(obj, 1, 2), (1, 2, ()))
    self.assertEqual(foo(obj, 1, 2, 3), (1, 2, (3,)))
    self.assertEqual(foo(obj, 1, 2, 3, 4), (1, 2, (3, 4)))

    @njit
    def bar(obj):
        return (obj.method_stararg(1, 2), obj.method_stararg(1, 2, 3), obj.method_stararg(1, 2, 3, 4))
    self.assertEqual(bar(obj), ((1, 2, ()), (1, 2, (3,)), (1, 2, (3, 4))))
    self.assertEqual(foo(obj, 1, 2, (3,)), (1, 2, ((3,),)))
    self.assertEqual(foo(obj, 1, 2, (3, 4)), (1, 2, ((3, 4),)))
    self.assertEqual(foo(obj, 1, 2, (3, (4, 5))), (1, 2, ((3, (4, 5)),)))