import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
@run_in_parallel()
def test_keypoints_censure_moon_image_octagon():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for Octagon filter."""
    detector = CENSURE(mode='octagon')
    detector.detect(rescale(img, 0.25, anti_aliasing=False, mode='constant'))
    expected_keypoints = np.array([[23, 27], [29, 89], [31, 87], [106, 59], [111, 67]])
    expected_scales = np.array([3, 2, 5, 2, 4])
    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)