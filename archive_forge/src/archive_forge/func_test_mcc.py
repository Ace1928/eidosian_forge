import numpy as np
import pytest
from skimage.measure import (
def test_mcc():
    img1 = np.array([[j for j in range(4)] for i in range(4)])
    mask = np.array([[i <= 1 for j in range(4)] for i in range(4)])
    assert manders_coloc_coeff(img1, mask) == 0.5
    img_negativeint = np.where(img1 == 1, -1, img1)
    img_negativefloat = img_negativeint / 2.0
    with pytest.raises(ValueError):
        manders_coloc_coeff(img_negativeint, mask)
    with pytest.raises(ValueError):
        manders_coloc_coeff(img_negativefloat, mask)