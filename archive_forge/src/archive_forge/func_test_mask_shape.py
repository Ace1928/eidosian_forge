import numpy as np
from skimage.measure import find_contours
from skimage._shared.testing import assert_array_equal
import pytest
@pytest.mark.parametrize('level', [0, None])
def test_mask_shape(level):
    bad_mask = np.ones((8, 7), dtype=bool)
    with pytest.raises(ValueError, match='shape'):
        find_contours(a, level, mask=bad_mask)