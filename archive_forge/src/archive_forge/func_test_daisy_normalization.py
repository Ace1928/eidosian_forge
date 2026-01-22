import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_daisy_normalization():
    img = img_as_float(data.astronaut()[:64, :64].mean(axis=2))
    descs = daisy(img, normalization='l1')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(np.sum(descs[i, j, :]), 1)
    descs_ = daisy(img)
    assert_almost_equal(descs, descs_)
    descs = daisy(img, normalization='l2')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(sqrt(np.sum(descs[i, j, :] ** 2)), 1)
    orientations = 8
    descs = daisy(img, orientations=orientations, normalization='daisy')
    desc_dims = descs.shape[2]
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            for k in range(0, desc_dims, orientations):
                assert_almost_equal(sqrt(np.sum(descs[i, j, k:k + orientations] ** 2)), 1)
    img = np.zeros((50, 50))
    descs = daisy(img, normalization='off')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(np.sum(descs[i, j, :]), 0)
    with pytest.raises(ValueError):
        daisy(img, normalization='does_not_exist')