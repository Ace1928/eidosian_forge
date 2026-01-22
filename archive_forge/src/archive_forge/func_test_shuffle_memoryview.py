import warnings
import pytest
import numpy as np
from numpy.testing import (
from numpy import random
import sys
def test_shuffle_memoryview(self):
    np.random.seed(self.seed)
    a = np.arange(5).data
    np.random.shuffle(a)
    assert_equal(np.asarray(a), [0, 1, 4, 3, 2])
    rng = np.random.RandomState(self.seed)
    rng.shuffle(a)
    assert_equal(np.asarray(a), [0, 1, 2, 3, 4])
    rng = np.random.default_rng(self.seed)
    rng.shuffle(a)
    assert_equal(np.asarray(a), [4, 1, 0, 3, 2])