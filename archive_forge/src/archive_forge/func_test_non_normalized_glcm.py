import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_non_normalized_glcm(self):
    img = (np.random.random((100, 100)) * 8).astype(np.uint8)
    p = graycomatrix(img, [1, 2, 4, 5], [0, 0.25, 1, 1.5], levels=8)
    np.testing.assert_(np.max(graycoprops(p, 'correlation')) < 1.0)