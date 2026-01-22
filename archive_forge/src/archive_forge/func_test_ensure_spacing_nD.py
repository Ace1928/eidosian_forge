import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
@pytest.mark.parametrize('ndim', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('size', [2, 10, None])
def test_ensure_spacing_nD(ndim, size):
    coord = np.ones((5, ndim))
    expected = np.ones((1, ndim))
    assert np.array_equal(ensure_spacing(coord, min_split_size=size), expected)