import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_width_patch():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        extract_patches_2d(x, (4, 1))
    with pytest.raises(ValueError):
        extract_patches_2d(x, (1, 4))