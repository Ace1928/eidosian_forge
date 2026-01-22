from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_singleton_dim():
    """Ensure images where size of a given dimension is 1 work correctly."""
    image = np.random.rand(1, 1000)
    noisy = random_noise(image, mode='salt', amount=0.1, rng=42)
    tolerance = 0.05
    assert abs(np.average(noisy == 1) - 0.1) <= tolerance