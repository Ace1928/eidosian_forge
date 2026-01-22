import numpy as np
import pytest
from skimage import data
from skimage.restoration._rolling_ball import rolling_ball
from skimage.restoration._rolling_ball import ellipsoid_kernel
def test_nan_const():
    img = 123 * np.ones((100, 100), dtype=float)
    img[20, 20] = np.nan
    img[50, 53] = np.nan
    kernel_shape = (10, 10)
    x = np.arange(-kernel_shape[1] // 2, kernel_shape[1] // 2 + 1)[np.newaxis, :]
    y = np.arange(-kernel_shape[0] // 2, kernel_shape[0] // 2 + 1)[:, np.newaxis]
    expected_img = np.zeros_like(img)
    expected_img[y + 20, x + 20] = np.nan
    expected_img[y + 50, x + 53] = np.nan
    kernel = ellipsoid_kernel(kernel_shape, 100)
    background = rolling_ball(img, kernel=kernel, nansafe=True)
    assert np.allclose(img - background, expected_img, equal_nan=True)