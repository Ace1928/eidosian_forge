from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_scalar_empty_dataset(self):
    """Test the scalar's cov factor for an empty array."""
    with pytest.raises(ValueError):
        mlab.GaussianKDE([], bw_method=5)