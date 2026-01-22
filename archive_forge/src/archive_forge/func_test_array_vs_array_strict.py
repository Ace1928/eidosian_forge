import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_vs_array_strict(self):
    """Test comparing two arrays with strict option."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert_array_equal(a, b, strict=True)