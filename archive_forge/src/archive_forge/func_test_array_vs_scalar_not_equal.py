import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_vs_scalar_not_equal(self):
    """Test comparing an array with a scalar when not all values equal."""
    a = np.array([1.0, 2.0, 3.0])
    b = 1.0
    self._test_not_equal(a, b)