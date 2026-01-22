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
def test_overload_method_kwargs(self):

    @overload_method(types.Array, 'foo')
    def fooimpl(arr, a_kwarg=10):

        def impl(arr, a_kwarg=10):
            return a_kwarg
        return impl

    @njit
    def bar(A):
        return (A.foo(), A.foo(20), A.foo(a_kwarg=30))
    Z = np.arange(5)
    self.assertEqual(bar(Z), (10, 20, 30))