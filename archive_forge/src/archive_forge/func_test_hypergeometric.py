import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_hypergeometric(self):
    ngood = [1]
    nbad = [2]
    nsample = [2]
    bad_ngood = [-1]
    bad_nbad = [-2]
    bad_nsample_one = [0]
    bad_nsample_two = [4]
    hypergeom = random.hypergeometric
    desired = np.array([1, 1, 1])
    self.set_seed()
    actual = hypergeom(ngood * 3, nbad, nsample)
    assert_array_equal(actual, desired)
    assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
    assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
    assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
    assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)
    self.set_seed()
    actual = hypergeom(ngood, nbad * 3, nsample)
    assert_array_equal(actual, desired)
    assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
    assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
    assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
    assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)
    self.set_seed()
    actual = hypergeom(ngood, nbad, nsample * 3)
    assert_array_equal(actual, desired)
    assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
    assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
    assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
    assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)
    assert_raises(ValueError, hypergeom, -1, 10, 20)
    assert_raises(ValueError, hypergeom, 10, -1, 20)
    assert_raises(ValueError, hypergeom, 10, 10, 0)
    assert_raises(ValueError, hypergeom, 10, 10, 25)