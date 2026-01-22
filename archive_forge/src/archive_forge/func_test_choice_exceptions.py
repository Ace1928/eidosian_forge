import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_exceptions(self):
    sample = random.choice
    assert_raises(ValueError, sample, -1, 3)
    assert_raises(ValueError, sample, 3.0, 3)
    assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
    assert_raises(ValueError, sample, [], 3)
    assert_raises(ValueError, sample, [1, 2, 3, 4], 3, p=[[0.25, 0.25], [0.25, 0.25]])
    assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
    assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
    assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
    assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
    assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
    assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
    assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
    assert_raises(ValueError, sample, [1, 2, 3], 2, replace=False, p=[1, 0, 0])