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
def test_caching_overload_objmode(self):
    cache_dir = temp_directory(self.__class__.__name__)
    with override_config('CACHE_DIR', cache_dir):

        def realwork(x):
            arr = np.arange(x) / x
            return np.linalg.norm(arr)

        def python_code(x):
            return realwork(x)

        @overload(with_objmode_cache_ov_example)
        def _ov_with_objmode_cache_ov_example(x):

            def impl(x):
                with objmode(y='float64'):
                    y = python_code(x)
                return y
            return impl

        @njit(cache=True)
        def testcase(x):
            return with_objmode_cache_ov_example(x)
        expect = realwork(123)
        got = testcase(123)
        self.assertEqual(got, expect)
        testcase_cached = njit(cache=True)(testcase.py_func)
        got = testcase_cached(123)
        self.assertEqual(got, expect)