import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_shuffle_masked(self):
    a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
    b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
    a_orig = a.copy()
    b_orig = b.copy()
    for i in range(50):
        random.shuffle(a)
        assert_equal(sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
        random.shuffle(b)
        assert_equal(sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_shuffle_invalid_objects(self):
        x = np.array(3)
        assert_raises(TypeError, random.shuffle, x)