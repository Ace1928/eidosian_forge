import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_choice_multidimensional_default_axis(self):
    random = Generator(MT19937(self.seed))
    actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
    desired = np.array([[0, 1], [0, 1], [4, 5]])
    assert_array_equal(actual, desired)