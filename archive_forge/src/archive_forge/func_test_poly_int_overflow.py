import numpy as np
from numpy.testing import (
import pytest
def test_poly_int_overflow(self):
    """
        Regression test for gh-5096.
        """
    v = np.arange(1, 21)
    assert_almost_equal(np.poly(v), np.poly(np.diag(v)))