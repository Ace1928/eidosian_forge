import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_non_finite_errors():
    lp = _LPProblem(c=[1, 2], A_ub=np.array([[1, 1], [2, 2]]), b_ub=np.array([1, 1]), A_eq=np.array([[1, 1], [2, 2]]), b_eq=np.array([1, 1]), bounds=[(0, 1)])
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, None]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.inf, 0]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, -np.inf]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.nan, 0]))
    assert_raises(ValueError, _clean_inputs, lp._replace(A_ub=[[1, 2], [None, 1]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_ub=[np.inf, 1]))
    assert_raises(ValueError, _clean_inputs, lp._replace(A_eq=[[1, 2], [1, -np.inf]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_eq=[1, np.nan]))