import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_keypoints_censure_color_image_unsupported_error():
    """Censure keypoints can be extracted from gray-scale images only."""
    with testing.raises(ValueError):
        CENSURE().detect(np.zeros((20, 20, 3)))