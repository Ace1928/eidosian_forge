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
def test_t_eval_early_event():

    def early_event(t, y):
        return t - 7
    early_event.terminal = True
    rtol = 0.001
    atol = 1e-06
    y0 = [1 / 3, 2 / 9]
    t_span = [5, 9]
    t_eval = np.linspace(7.5, 9, 16)
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'The following arguments have no effect for a chosen solver: `jac`')
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol, method=method, t_eval=t_eval, events=early_event, jac=jac_rational)
        assert res.success
        assert res.message == 'A termination event occurred.'
        assert res.status == 1
        assert not res.t and (not res.y)
        assert len(res.t_events) == 1
        assert res.t_events[0].size == 1
        assert res.t_events[0][0] == 7