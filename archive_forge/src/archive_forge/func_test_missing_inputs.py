import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_missing_inputs():
    c = [1, 2]
    A_ub = np.array([[1, 1], [2, 2]])
    b_ub = np.array([1, 1])
    A_eq = np.array([[1, 1], [2, 2]])
    b_eq = np.array([1, 1])
    assert_raises(TypeError, _clean_inputs)
    assert_raises(TypeError, _clean_inputs, _LPProblem(c=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub, b_ub=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=None, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq, b_eq=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_eq=b_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=None, b_eq=b_eq))