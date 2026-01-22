import numpy as np
import pytest
from skimage import data
from skimage.restoration._rolling_ball import rolling_ball
from skimage.restoration._rolling_ball import ellipsoid_kernel
@pytest.mark.parametrize('radius', [1, 2.5, 10.346, 50])
def test_const_image(radius):
    img = 23 * np.ones((100, 100), dtype=np.uint8)
    background = rolling_ball(img, radius=radius)
    assert np.allclose(img - background, np.zeros_like(img))