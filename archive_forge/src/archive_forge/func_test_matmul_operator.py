import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
@skip_unless_scipy
def test_matmul_operator(self):

    class ArrayAt:

        def __init__(self, array):
            self.arr = array

        def __matmul__(self, other):
            return self.arr @ other.arr

        def __rmatmul__(self, other):
            return other.arr @ self.arr

        def __imatmul__(self, other):
            self.arr = self.arr @ other.arr
            return self

    class ArrayNoAt:

        def __init__(self, array):
            self.arr = array
    n = 3
    np.random.seed(1)
    vec = np.random.random(size=(n,))
    mat = np.random.random(size=(n, n))
    vector_noat = ArrayNoAt(vec)
    vector_at = ArrayAt(vec)
    jit_vector_noat = jitclass(ArrayNoAt, spec={'arr': float64[::1]})(vec)
    jit_vector_at = jitclass(ArrayAt, spec={'arr': float64[::1]})(vec)
    matrix_noat = ArrayNoAt(mat)
    matrix_at = ArrayAt(mat)
    jit_matrix_noat = jitclass(ArrayNoAt, spec={'arr': float64[:, ::1]})(mat)
    jit_matrix_at = jitclass(ArrayAt, spec={'arr': float64[:, ::1]})(mat)
    np.testing.assert_allclose(vector_at @ vector_noat, jit_vector_at @ jit_vector_noat)
    np.testing.assert_allclose(vector_at @ matrix_noat, jit_vector_at @ jit_matrix_noat)
    np.testing.assert_allclose(matrix_at @ vector_noat, jit_matrix_at @ jit_vector_noat)
    np.testing.assert_allclose(matrix_at @ matrix_noat, jit_matrix_at @ jit_matrix_noat)
    np.testing.assert_allclose(vector_noat @ vector_at, jit_vector_noat @ jit_vector_at)
    np.testing.assert_allclose(vector_noat @ matrix_at, jit_vector_noat @ jit_matrix_at)
    np.testing.assert_allclose(matrix_noat @ vector_at, jit_matrix_noat @ jit_vector_at)
    np.testing.assert_allclose(matrix_noat @ matrix_at, jit_matrix_noat @ jit_matrix_at)
    vector_at @= matrix_noat
    matrix_at @= matrix_noat
    jit_vector_at @= jit_matrix_noat
    jit_matrix_at @= jit_matrix_noat
    np.testing.assert_allclose(vector_at.arr, jit_vector_at.arr)
    np.testing.assert_allclose(matrix_at.arr, jit_matrix_at.arr)