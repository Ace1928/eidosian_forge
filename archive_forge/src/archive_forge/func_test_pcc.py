import numpy as np
import pytest
from skimage.measure import (
def test_pcc():
    img1 = np.array([[i + j for j in range(4)] for i in range(4)])
    np.testing.assert_almost_equal(pearson_corr_coeff(img1, img1), (1.0, 0.0), decimal=14)
    img2 = np.where(img1 <= 2, 0, img1)
    np.testing.assert_almost_equal(pearson_corr_coeff(img1, img2), (0.944911182523068, 3.5667540654536515e-08))
    roi = np.where(img1 <= 2, 0, 1)
    np.testing.assert_almost_equal(pearson_corr_coeff(img1, img1, roi), pearson_corr_coeff(img1, img2, roi))