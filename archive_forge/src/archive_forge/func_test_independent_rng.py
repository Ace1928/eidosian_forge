import pytest
import copy
import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing
def test_independent_rng():
    img = np.zeros((100, 100), dtype=int)
    keypoints = np.array([[1, 1], [20, 20], [50, 50], [80, 80]])
    rng = np.random.default_rng()
    extractor = BRIEF(patch_size=41, rng=rng)
    x = copy.deepcopy(extractor.rng).random()
    rng.random()
    extractor.extract(img, keypoints)
    z = copy.deepcopy(extractor.rng).random()
    assert x == z