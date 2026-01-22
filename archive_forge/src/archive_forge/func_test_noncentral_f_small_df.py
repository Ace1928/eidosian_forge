import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_noncentral_f_small_df(self):
    self.set_seed()
    desired = np.array([6.869638627492048, 0.785880199263955])
    actual = random.noncentral_f(0.9, 0.9, 2, size=2)
    assert_array_almost_equal(actual, desired, decimal=14)