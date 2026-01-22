import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permutation_exceptions(self):
    random = Generator(MT19937(self.seed))
    arr = np.arange(10)
    assert_raises(np.AxisError, random.permutation, arr, 1)
    arr = np.arange(9).reshape((3, 3))
    assert_raises(np.AxisError, random.permutation, arr, 3)
    assert_raises(TypeError, random.permutation, arr, slice(1, 2, None))