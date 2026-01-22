import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_rank1_eq(self):
    """Test two equal array of rank 1 are found equal."""
    a = np.array([1, 2])
    b = np.array([1, 2])
    self._test_equal(a, b)