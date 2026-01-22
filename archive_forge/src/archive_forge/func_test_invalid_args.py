import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_invalid_args(self):
    with assert_raises(ValueError, match='must be of same length'):
        firwin2(50, [0, 0.5, 1], [0.0, 1.0])
    with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
        firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
    with assert_raises(ValueError, match='must be nondecreasing'):
        firwin2(50, [0, 0.5, 0.4, 1.0], [0, 0.25, 0.5, 1.0])
    with assert_raises(ValueError, match='must not occur more than twice'):
        firwin2(50, [0, 0.1, 0.1, 0.1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
    with assert_raises(ValueError, match='start with 0'):
        firwin2(50, [0.5, 1.0], [0.0, 1.0])
    with assert_raises(ValueError, match='end with fs/2'):
        firwin2(50, [0.0, 0.5], [0.0, 1.0])
    with assert_raises(ValueError, match='0 must not be repeated'):
        firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
    with assert_raises(ValueError, match='fs/2 must not be repeated'):
        firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
    with assert_raises(ValueError, match='cannot contain numbers that are too close'):
        firwin2(50, [0.0, 0.5 - np.finfo(float).eps * 0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0])
    with assert_raises(ValueError, match='Type II filter'):
        firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    with assert_raises(ValueError, match='Type III filter'):
        firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
    with assert_raises(ValueError, match='Type III filter'):
        firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
    with assert_raises(ValueError, match='Type III filter'):
        firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)
    with assert_raises(ValueError, match='Type IV filter'):
        firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)