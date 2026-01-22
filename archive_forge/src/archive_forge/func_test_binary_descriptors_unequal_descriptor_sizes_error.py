import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import BRIEF, match_descriptors, corner_peaks, corner_harris
from skimage._shared import testing
def test_binary_descriptors_unequal_descriptor_sizes_error():
    """Sizes of descriptors of keypoints to be matched should be equal."""
    descs1 = np.array([[True, True, False, True], [False, True, False, True]])
    descs2 = np.array([[True, False, False, True, False], [False, True, True, True, False]])
    with testing.raises(ValueError):
        match_descriptors(descs1, descs2)