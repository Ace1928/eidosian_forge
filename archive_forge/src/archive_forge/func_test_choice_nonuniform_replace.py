import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_nonuniform_replace(self):
    random.seed(self.seed)
    actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
    desired = np.array([1, 1, 2, 2])
    assert_array_equal(actual, desired)