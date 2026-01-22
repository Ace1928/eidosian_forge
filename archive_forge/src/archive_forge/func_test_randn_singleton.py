import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_randn_singleton(self):
    random.seed(self.seed)
    actual = random.randn()
    desired = np.array(1.3401634577186312)
    assert_array_almost_equal(actual, desired, decimal=15)