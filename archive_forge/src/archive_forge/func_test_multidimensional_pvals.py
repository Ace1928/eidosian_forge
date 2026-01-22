import warnings
import pytest
import numpy as np
from numpy.testing import (
from numpy import random
import sys
def test_multidimensional_pvals(self):
    assert_raises(ValueError, np.random.multinomial, 10, [[0, 1]])
    assert_raises(ValueError, np.random.multinomial, 10, [[0], [1]])
    assert_raises(ValueError, np.random.multinomial, 10, [[[0], [1]], [[1], [0]]])
    assert_raises(ValueError, np.random.multinomial, 10, np.array([[0, 1], [1, 0]]))