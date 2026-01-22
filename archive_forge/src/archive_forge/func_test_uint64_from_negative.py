import pytest
import numpy as np
from numpy.testing import (
def test_uint64_from_negative(self):
    with pytest.warns(DeprecationWarning):
        assert_equal(np.uint64(-2), np.uint64(18446744073709551614))