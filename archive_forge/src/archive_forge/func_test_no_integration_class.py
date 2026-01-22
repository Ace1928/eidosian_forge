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
def test_no_integration_class():
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        solver = method(lambda t, y: -y, 0.0, [10.0, 0.0], 0.0)
        solver.step()
        assert_equal(solver.status, 'finished')
        sol = solver.dense_output()
        assert_equal(sol(0.0), [10.0, 0.0])
        assert_equal(sol([0, 1, 2]), [[10, 10, 10], [0, 0, 0]])
        solver = method(lambda t, y: -y, 0.0, [], np.inf)
        solver.step()
        assert_equal(solver.status, 'finished')
        sol = solver.dense_output()
        assert_equal(sol(100.0), [])
        assert_equal(sol([0, 1, 2]), np.empty((0, 3)))