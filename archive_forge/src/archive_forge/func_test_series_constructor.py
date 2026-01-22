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
def test_series_constructor(self):
    i = Index(np.int32([42, 8, -5]))
    d = np.float64([1.5, 4.0, 2.5])
    cfunc = jit(nopython=True)(make_series_usecase)
    got = cfunc(d, i)
    self.assertIsInstance(got, Series)
    self.assertIsInstance(got._index, Index)
    self.assertIs(got._index._data, i._data)
    self.assertIs(got._values, d)