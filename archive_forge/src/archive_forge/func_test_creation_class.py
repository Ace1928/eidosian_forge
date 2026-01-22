import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_creation_class(self):
    arr1 = np.array([1.0, 2.0, 3.0], dtype=SF)
    assert arr1.dtype == SF(1.0)
    arr2 = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
    assert_array_equal(arr1.view(np.float64), arr2.view(np.float64))