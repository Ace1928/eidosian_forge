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
def test_unbounded_below_no_presolve_corrected(self):
    c = [1]
    bounds = [(None, 1)]
    o = {key: self.options[key] for key in self.options}
    o['presolve'] = False
    res = linprog(c=c, bounds=bounds, method=self.method, options=o)
    if self.method == 'revised simplex':
        assert_equal(res.status, 5)
    else:
        _assert_unbounded(res)