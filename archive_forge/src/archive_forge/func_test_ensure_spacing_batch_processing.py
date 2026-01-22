import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
@pytest.mark.parametrize('p', [1, 2, np.inf])
@pytest.mark.parametrize('size', [50, 100, None])
def test_ensure_spacing_batch_processing(p, size):
    coord = np.random.randn(100, 2)
    spacing = np.median(pdist(coord, metric=minkowski, p=p))
    expected = ensure_spacing(coord, spacing=spacing, p_norm=p)
    assert np.array_equal(ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size), expected)