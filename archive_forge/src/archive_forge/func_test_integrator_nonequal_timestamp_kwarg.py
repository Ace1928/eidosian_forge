from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_integrator_nonequal_timestamp_kwarg(self):
    t = np.array([0.0, 1.0, 1.0, 1.1, 1.1, 2.0])
    u = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    system = ([1.0], [1.0, 0.0])
    tout, y, x = self.func(system, u, t, hmax=0.01)
    expected_x = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    assert_almost_equal(x, expected_x)