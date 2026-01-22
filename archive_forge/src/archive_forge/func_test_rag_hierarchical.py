import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_rag_hierarchical():
    img = np.zeros((8, 8, 3), dtype='uint8')
    labels = np.zeros((8, 8), dtype='uint8')
    img[:, :, :] = 31
    labels[:, :] = 1
    img[0:4, 0:4, :] = (10, 10, 10)
    labels[0:4, 0:4] = 2
    img[4:, 0:4, :] = (20, 20, 20)
    labels[4:, 0:4] = 3
    g = graph.rag_mean_color(img, labels)
    g2 = g.copy()
    thresh = 20
    result = merge_hierarchical_mean_color(labels, g, thresh)
    assert np.all(result[:, :4] == result[0, 0])
    assert np.all(result[:, 4:] == result[-1, -1])
    result = merge_hierarchical_mean_color(labels, g2, thresh, in_place_merge=True)
    assert np.all(result[:, :4] == result[0, 0])
    assert np.all(result[:, 4:] == result[-1, -1])
    result = graph.cut_threshold(labels, g, thresh)
    assert np.all(result == result[0, 0])