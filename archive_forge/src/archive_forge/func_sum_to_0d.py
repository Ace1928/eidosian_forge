import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def sum_to_0d(x):
    """ Sum x, returning a 0d array of the same class """
    assert_equal(x.ndim, 1)
    return np.squeeze(np.sum(x, keepdims=True))