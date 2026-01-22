import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_bounds_fixed(self):
    do_presolve = self.options.get('presolve', True)
    res = linprog([1], bounds=(1, 1), method=self.method, options=self.options)
    _assert_success(res, 1, 1)
    if do_presolve:
        assert_equal(res.nit, 0)
    res = linprog([1, 2, 3], bounds=[(5, 5), (-1, -1), (3, 3)], method=self.method, options=self.options)
    _assert_success(res, 12, [5, -1, 3])
    if do_presolve:
        assert_equal(res.nit, 0)
    res = linprog([1, 1], bounds=[(1, 1), (1, 3)], method=self.method, options=self.options)
    _assert_success(res, 2, [1, 1])
    if do_presolve:
        assert_equal(res.nit, 0)
    res = linprog([1, 1, 2], A_eq=[[1, 0, 0], [0, 1, 0]], b_eq=[1, 7], bounds=[(-5, 5), (0, 10), (3.5, 3.5)], method=self.method, options=self.options)
    _assert_success(res, 15, [1, 7, 3.5])
    if do_presolve:
        assert_equal(res.nit, 0)