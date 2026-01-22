import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_vonmises(self):
    mu = [2]
    kappa = [1]
    bad_kappa = [-1]
    vonmises = random.vonmises
    desired = np.array([2.9883443664201312, -2.7064099483995943, -1.8672476700665914])
    self.set_seed()
    actual = vonmises(mu * 3, kappa)
    assert_array_almost_equal(actual, desired, decimal=14)
    assert_raises(ValueError, vonmises, mu * 3, bad_kappa)
    self.set_seed()
    actual = vonmises(mu, kappa * 3)
    assert_array_almost_equal(actual, desired, decimal=14)
    assert_raises(ValueError, vonmises, mu, bad_kappa * 3)