import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_keypoints_censure_moon_image_star():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for STAR filter."""
    detector = CENSURE(mode='star')
    detector.detect(rescale(img, 0.25, anti_aliasing=False, mode='constant'))
    expected_keypoints = np.array([[23, 27], [29, 89], [30, 86], [107, 59], [109, 64], [111, 67], [113, 70]])
    expected_scales = np.array([3, 2, 4, 2, 5, 3, 2])
    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)