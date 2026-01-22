import numpy as np
import pytest
from skimage.metrics import (
from skimage._shared.testing import (
def test_vi_ignore_labels():
    im1 = np.array([[1, 0], [2, 3]], dtype='uint8')
    im2 = np.array([[1, 1], [1, 0]], dtype='uint8')
    false_splits, false_merges = variation_of_information(im1, im2, ignore_labels=[0])
    assert (false_splits, false_merges) == (0, 2 / 3)