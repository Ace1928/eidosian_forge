import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_noninteger(self):
    random.seed(self.seed)
    actual = random.choice(['a', 'b', 'c', 'd'], 4)
    desired = np.array(['c', 'd', 'c', 'd'])
    assert_array_equal(actual, desired)