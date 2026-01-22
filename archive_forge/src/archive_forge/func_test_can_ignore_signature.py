import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_can_ignore_signature(self):
    mat = np.arange(12).reshape((2, 3, 2))
    single_vec = np.arange(2)
    col_vec = single_vec[:, np.newaxis]
    col_vec_array = np.arange(8).reshape((2, 2, 2, 1)) + 1
    mm_col_vec = umt.matrix_multiply(mat, col_vec)
    matmul_col_vec = umt.matmul(mat, col_vec)
    assert_array_equal(matmul_col_vec, mm_col_vec)
    assert_raises(ValueError, umt.matrix_multiply, mat, single_vec)
    matmul_col = umt.matmul(mat, single_vec)
    assert_array_equal(matmul_col, mm_col_vec.squeeze())
    mm_col_vec = umt.matrix_multiply(mat, col_vec_array)
    matmul_col_vec = umt.matmul(mat, col_vec_array)
    assert_array_equal(matmul_col_vec, mm_col_vec)
    single_vec = np.arange(3)
    row_vec = single_vec[np.newaxis, :]
    row_vec_array = np.arange(24).reshape((4, 2, 1, 1, 3)) + 1
    mm_row_vec = umt.matrix_multiply(row_vec, mat)
    matmul_row_vec = umt.matmul(row_vec, mat)
    assert_array_equal(matmul_row_vec, mm_row_vec)
    assert_raises(ValueError, umt.matrix_multiply, single_vec, mat)
    matmul_row = umt.matmul(single_vec, mat)
    assert_array_equal(matmul_row, mm_row_vec.squeeze())
    mm_row_vec = umt.matrix_multiply(row_vec_array, mat)
    matmul_row_vec = umt.matmul(row_vec_array, mat)
    assert_array_equal(matmul_row_vec, mm_row_vec)
    col_vec = row_vec.T
    col_vec_array = row_vec_array.swapaxes(-2, -1)
    mm_row_col_vec = umt.matrix_multiply(row_vec, col_vec)
    matmul_row_col_vec = umt.matmul(row_vec, col_vec)
    assert_array_equal(matmul_row_col_vec, mm_row_col_vec)
    assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec)
    matmul_row_col = umt.matmul(single_vec, single_vec)
    assert_array_equal(matmul_row_col, mm_row_col_vec.squeeze())
    mm_row_col_array = umt.matrix_multiply(row_vec_array, col_vec_array)
    matmul_row_col_array = umt.matmul(row_vec_array, col_vec_array)
    assert_array_equal(matmul_row_col_array, mm_row_col_array)
    out = np.zeros_like(mm_row_col_array)
    out = umt.matrix_multiply(row_vec_array, col_vec_array, out=out)
    assert_array_equal(out, mm_row_col_array)
    out[:] = 0
    out = umt.matmul(row_vec_array, col_vec_array, out=out)
    assert_array_equal(out, mm_row_col_array)
    out = np.zeros_like(mm_row_col_vec)
    assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec, out)
    out = umt.matmul(single_vec, single_vec, out)
    assert_array_equal(out, mm_row_col_vec.squeeze())