import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_uniform_replace(self):
    random.seed(self.seed)
    actual = random.choice(4, 4)
    desired = np.array([2, 3, 2, 3])
    assert_array_equal(actual, desired)