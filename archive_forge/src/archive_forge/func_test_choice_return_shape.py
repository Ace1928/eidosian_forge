import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_return_shape(self):
    p = [0.1, 0.9]
    assert_(np.isscalar(random.choice(2, replace=True)))
    assert_(np.isscalar(random.choice(2, replace=False)))
    assert_(np.isscalar(random.choice(2, replace=True, p=p)))
    assert_(np.isscalar(random.choice(2, replace=False, p=p)))
    assert_(np.isscalar(random.choice([1, 2], replace=True)))
    assert_(random.choice([None], replace=True) is None)
    a = np.array([1, 2])
    arr = np.empty(1, dtype=object)
    arr[0] = a
    assert_(random.choice(arr, replace=True) is a)
    s = tuple()
    assert_(not np.isscalar(random.choice(2, s, replace=True)))
    assert_(not np.isscalar(random.choice(2, s, replace=False)))
    assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
    assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
    assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
    assert_(random.choice([None], s, replace=True).ndim == 0)
    a = np.array([1, 2])
    arr = np.empty(1, dtype=object)
    arr[0] = a
    assert_(random.choice(arr, s, replace=True).item() is a)
    s = (2, 3)
    p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
    assert_equal(random.choice(6, s, replace=True).shape, s)
    assert_equal(random.choice(6, s, replace=False).shape, s)
    assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
    assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
    assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)
    assert_equal(random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
    assert_equal(random.randint(0, -10, size=0).shape, (0,))
    assert_equal(random.randint(10, 10, size=0).shape, (0,))
    assert_equal(random.choice(0, size=0).shape, (0,))
    assert_equal(random.choice([], size=(0,)).shape, (0,))
    assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape, (3, 0, 4))
    assert_raises(ValueError, random.choice, [], 10)