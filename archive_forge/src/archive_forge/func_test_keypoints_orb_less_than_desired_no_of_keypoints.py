import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64'])
def test_keypoints_orb_less_than_desired_no_of_keypoints(dtype):
    _img = _convert(img, dtype)
    detector_extractor = ORB(n_keypoints=15, fast_n=12, fast_threshold=0.33, downscale=2, n_scales=2)
    detector_extractor.detect(_img)
    exp_rows = np.array([108.0, 203.0, 140.0, 65.0, 58.0])
    exp_cols = np.array([293.0, 267.0, 202.0, 130.0, 291.0])
    exp_scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    exp_orientations = np.array([151.93906, -56.90052, -79.46341, -59.42996, -158.26941])
    exp_response = np.array([-0.1764169, 0.2652126, -0.0324343, 0.0400902, 0.2667641])
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses)
    assert_almost_equal(exp_orientations, np.rad2deg(detector_extractor.orientations), 3)
    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])