from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_missing_D(self):
    A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
    assert_equal(D.shape[0], C.shape[0])
    assert_equal(D.shape[1], B.shape[1])
    assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))