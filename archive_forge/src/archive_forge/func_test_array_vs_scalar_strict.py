import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_vs_scalar_strict(self):
    """Test comparing an array with a scalar with strict option."""
    a = np.array([1.0, 1.0, 1.0])
    b = 1.0
    with pytest.raises(AssertionError):
        assert_array_equal(a, b, strict=True)