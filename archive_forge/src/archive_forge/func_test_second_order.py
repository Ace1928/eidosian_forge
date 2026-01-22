from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_second_order(self):
    system = ([1.0], [1.0, 2.0, 1.0])
    tout, y = self.func(system)
    expected_y = 1 - (1 + tout) * np.exp(-tout)
    assert_almost_equal(y, expected_y)