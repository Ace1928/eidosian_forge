import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_axes_invalid():
    data = np.empty((2, 3))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 3)], axes=[2])