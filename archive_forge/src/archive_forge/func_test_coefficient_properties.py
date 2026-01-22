import pytest
from numpy.testing import assert_allclose, assert_
import numpy as np
from scipy.integrate import RK23, RK45, DOP853
from scipy.integrate._ivp import dop853_coefficients
@pytest.mark.parametrize('solver', [RK23, RK45, DOP853])
def test_coefficient_properties(solver):
    assert_allclose(np.sum(solver.B), 1, rtol=1e-15)
    assert_allclose(np.sum(solver.A, axis=1), solver.C, rtol=1e-14)