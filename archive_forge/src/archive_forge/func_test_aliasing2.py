import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_aliasing2():
    """
    Similar purpose as `test_aliasing` above.
    """
    lp = _LPProblem(c=np.array([1, 1]), A_ub=np.array([[1, 1], [2, 2]]), b_ub=np.array([[1], [1]]), A_eq=np.array([[1, 1]]), b_eq=np.array([1]), bounds=[(-np.inf, np.inf), (None, 1)])
    lp_copy = deepcopy(lp)
    _clean_inputs(lp)
    assert_allclose(lp.c, lp_copy.c, err_msg='c modified by _clean_inputs')
    assert_allclose(lp.A_ub, lp_copy.A_ub, err_msg='A_ub modified by _clean_inputs')
    assert_allclose(lp.b_ub, lp_copy.b_ub, err_msg='b_ub modified by _clean_inputs')
    assert_allclose(lp.A_eq, lp_copy.A_eq, err_msg='A_eq modified by _clean_inputs')
    assert_allclose(lp.b_eq, lp_copy.b_eq, err_msg='b_eq modified by _clean_inputs')
    assert_(lp.bounds == lp_copy.bounds, 'bounds modified by _clean_inputs')