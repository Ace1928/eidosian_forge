import numpy as np
import pytest
from skimage.measure import (
def test_intersection_coefficient():
    img1_mask = np.array([[j <= 1 for j in range(4)] for i in range(4)])
    img2_mask = np.array([[i <= 1 for j in range(4)] for i in range(4)])
    img3_mask = np.array([[1 for j in range(4)] for i in range(4)])
    assert intersection_coeff(img1_mask, img2_mask) == 0.5
    assert intersection_coeff(img1_mask, img3_mask) == 1