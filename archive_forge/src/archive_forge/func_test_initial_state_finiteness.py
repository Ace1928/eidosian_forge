from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
@pytest.mark.parametrize('f0_fill', [np.nan, np.inf])
def test_initial_state_finiteness(f0_fill):
    msg = 'All components of the initial state `y0` must be finite.'
    with pytest.raises(ValueError, match=msg):
        solve_ivp(fun_zero, [0, 10], np.full(3, f0_fill))