import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test__clean_inputs3():
    lp = _LPProblem(c=[[1, 2]], A_ub=np.random.rand(2, 2), b_ub=[[1], [2]], A_eq=np.random.rand(2, 2), b_eq=[[1], [2]], bounds=[(0, 1)])
    lp_cleaned = _clean_inputs(lp)
    assert_allclose(lp_cleaned.c, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_ub, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_eq, np.array([1, 2]))
    assert_equal(lp_cleaned.bounds, [(0, 1)] * 2)
    assert_(lp_cleaned.c.shape == (2,), '')
    assert_(lp_cleaned.b_ub.shape == (2,), '')
    assert_(lp_cleaned.b_eq.shape == (2,), '')