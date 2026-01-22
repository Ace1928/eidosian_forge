from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_zero_order_round_trip(self):
    tf = (2, 1)
    A, B, C, D = tf2ss(*tf)
    assert_allclose(A, [[0]], rtol=1e-13)
    assert_allclose(B, [[0]], rtol=1e-13)
    assert_allclose(C, [[0]], rtol=1e-13)
    assert_allclose(D, [[2]], rtol=1e-13)
    num, den = ss2tf(A, B, C, D)
    assert_allclose(num, [[2, 0]], rtol=1e-13)
    assert_allclose(den, [1, 0], rtol=1e-13)
    tf = ([[5], [2]], 1)
    A, B, C, D = tf2ss(*tf)
    assert_allclose(A, [[0]], rtol=1e-13)
    assert_allclose(B, [[0]], rtol=1e-13)
    assert_allclose(C, [[0], [0]], rtol=1e-13)
    assert_allclose(D, [[5], [2]], rtol=1e-13)
    num, den = ss2tf(A, B, C, D)
    assert_allclose(num, [[5, 0], [2, 0]], rtol=1e-13)
    assert_allclose(den, [1, 0], rtol=1e-13)