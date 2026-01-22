from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_speckle():
    seed = 42
    data = np.zeros((128, 128)) + 0.1
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.1, 0.02 ** 0.5, (128, 128))
    expected = np.clip(data + data * noise, 0, 1)
    data_speckle = random_noise(data, mode='speckle', rng=42, mean=0.1, var=0.02)
    assert_allclose(expected, data_speckle)