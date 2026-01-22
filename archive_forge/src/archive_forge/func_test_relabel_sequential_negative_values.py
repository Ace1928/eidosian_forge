import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_relabel_sequential_negative_values():
    ar = np.array([1, 1, 5, -5, 8, 99, 42, 0])
    with pytest.raises(ValueError):
        relabel_sequential(ar)