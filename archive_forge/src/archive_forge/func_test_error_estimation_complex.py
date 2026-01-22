import pytest
from numpy.testing import assert_allclose, assert_
import numpy as np
from scipy.integrate import RK23, RK45, DOP853
from scipy.integrate._ivp import dop853_coefficients
@pytest.mark.parametrize('solver_class', [RK23, RK45, DOP853])
def test_error_estimation_complex(solver_class):
    h = 0.2
    solver = solver_class(lambda t, y: 1j * y, 0, [1j], 1, first_step=h)
    solver.step()
    err_norm = solver._estimate_error_norm(solver.K, h, scale=[1])
    assert np.isrealobj(err_norm)