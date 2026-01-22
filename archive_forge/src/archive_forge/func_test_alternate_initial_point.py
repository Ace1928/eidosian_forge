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
def test_alternate_initial_point(self):
    A, b, c = lpgen_2d(20, 20)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'scipy.linalg.solve\nIll...')
        sup.filter(OptimizeWarning, 'Solving system with option...')
        sup.filter(LinAlgWarning, 'Ill-conditioned matrix...')
        res = linprog(c, A_ub=A, b_ub=b, method=self.method, options={'ip': True, 'disp': True})
    _assert_success(res, desired_fun=-64.049494229)