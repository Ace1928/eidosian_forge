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
def test_args_single_value():

    def fun_with_arg(t, y, a):
        return a * y
    message = "Supplied 'args' cannot be unpacked."
    with pytest.raises(TypeError, match=message):
        solve_ivp(fun_with_arg, (0, 0.1), [1], args=-1)
    sol = solve_ivp(fun_with_arg, (0, 0.1), [1], args=(-1,))
    assert_allclose(sol.y[0, -1], np.exp(-0.1))