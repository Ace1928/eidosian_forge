import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
@pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
def test_sfloat_from_float(self, scaling):
    a = np.array([1.0, 2.0, 3.0]).astype(dtype=SF(scaling))
    assert a.dtype.get_scaling() == scaling
    assert_array_equal(scaling * a.view(np.float64), [1.0, 2.0, 3.0])