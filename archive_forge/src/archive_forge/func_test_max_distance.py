import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import BRIEF, match_descriptors, corner_peaks, corner_harris
from skimage._shared import testing
def test_max_distance():
    descs1 = np.zeros((10, 128))
    descs2 = np.zeros((15, 128))
    descs1[0, :] = 1
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_distance=0.1, cross_check=False)
    assert len(matches) == 9
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_distance=np.sqrt(128.1), cross_check=False)
    assert len(matches) == 10
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_distance=0.1, cross_check=True)
    assert_equal(matches, [[1, 0]])
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_distance=np.sqrt(128.1), cross_check=True)
    assert_equal(matches, [[1, 0]])