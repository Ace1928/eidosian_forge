from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_zero_dimension_is_not_none1(self):
    B_ = np.zeros((2, 0))
    D_ = np.zeros((0, 0))
    A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
    assert_equal(A, self.A)
    assert_equal(B, B_)
    assert_equal(D, D_)
    assert_equal(C.shape[0], D_.shape[0])
    assert_equal(C.shape[1], self.A.shape[0])