from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_single_dataset_element(self):
    """Pass a single dataset element into the GaussianKDE class."""
    with pytest.raises(ValueError):
        mlab.GaussianKDE([42])