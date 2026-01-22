import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_aliasing():
    """
    Test for ensuring that no objects referred to by `lp` attributes,
    `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds`, have been modified
    by `_clean_inputs` as a side effect.
    """
    lp = _LPProblem(c=1, A_ub=[[1]], b_ub=[1], A_eq=[[1]], b_eq=[1], bounds=(-np.inf, np.inf))
    lp_copy = deepcopy(lp)
    _clean_inputs(lp)
    assert_(lp.c == lp_copy.c, 'c modified by _clean_inputs')
    assert_(lp.A_ub == lp_copy.A_ub, 'A_ub modified by _clean_inputs')
    assert_(lp.b_ub == lp_copy.b_ub, 'b_ub modified by _clean_inputs')
    assert_(lp.A_eq == lp_copy.A_eq, 'A_eq modified by _clean_inputs')
    assert_(lp.b_eq == lp_copy.b_eq, 'b_eq modified by _clean_inputs')
    assert_(lp.bounds == lp_copy.bounds, 'bounds modified by _clean_inputs')