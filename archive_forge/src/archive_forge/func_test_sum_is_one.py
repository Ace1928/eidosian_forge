import numpy as np
import scipy.special as sc
import pytest
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings
def test_sum_is_one(self):
    val = sc.bdtri([0, 1], 2, 0.5)
    actual = np.asarray([1 - 1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert_allclose(val, actual)