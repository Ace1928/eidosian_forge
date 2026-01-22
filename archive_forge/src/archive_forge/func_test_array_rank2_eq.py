import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_rank2_eq(self):
    """Test two equal array of rank 2 are found equal."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 4]])
    self._test_equal(a, b)