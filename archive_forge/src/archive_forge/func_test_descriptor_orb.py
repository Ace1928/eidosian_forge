import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert
@xfail(condition=arch32, reason='Known test failure on 32-bit platforms. See links for details: https://github.com/scikit-image/scikit-image/issues/3091 https://github.com/scikit-image/scikit-image/issues/2529')
def test_descriptor_orb():
    detector_extractor = ORB(fast_n=12, fast_threshold=0.2)
    exp_descriptors = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=bool)
    detector_extractor.detect(img)
    detector_extractor.extract(img, detector_extractor.keypoints, detector_extractor.scales, detector_extractor.orientations)
    assert_equal(exp_descriptors, detector_extractor.descriptors[100:120, 10:20])
    detector_extractor.detect_and_extract(img)
    assert_equal(exp_descriptors, detector_extractor.descriptors[100:120, 10:20])
    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.responses.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]