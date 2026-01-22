import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
@pytest.mark.parametrize('p', [1, 2, np.inf])
@pytest.mark.parametrize('size', [30, 50, None])
def test_ensure_spacing_trivial(p, size):
    assert ensure_spacing([], p_norm=p) == []
    coord = np.random.randn(1, 2)
    assert np.array_equal(coord, ensure_spacing(coord, p_norm=p, min_split_size=size))
    coord = np.random.randn(100, 2)
    assert np.array_equal(coord, ensure_spacing(coord, spacing=0, p_norm=p, min_split_size=size))
    spacing = pdist(coord, metric=minkowski, p=p).min() * 0.5
    out = ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size)
    assert np.array_equal(coord, out)