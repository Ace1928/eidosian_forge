import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_n_zero(self):
    zeros = np.zeros(2, dtype='int')
    for p in [0, 0.5, 1]:
        assert_(random.binomial(0, p) == 0)
        assert_array_equal(random.binomial(zeros, p), zeros)