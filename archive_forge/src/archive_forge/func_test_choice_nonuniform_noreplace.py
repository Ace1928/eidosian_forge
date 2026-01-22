import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_nonuniform_noreplace(self):
    random.seed(self.seed)
    actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
    desired = np.array([2, 3, 1])
    assert_array_equal(actual, desired)