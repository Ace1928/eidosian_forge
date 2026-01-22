from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_pepper():
    cam = img_as_float(camera())
    data_signed = cam * 2.0 - 1.0
    amount = 0.15
    cam_noisy = random_noise(cam, rng=42, mode='pepper', amount=amount)
    peppermask = cam != cam_noisy
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))
    proportion = float(peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 0.01
    assert abs(amount - proportion) <= tolerance
    orig_zeros = (data_signed == -1).sum()
    cam_noisy_signed = random_noise(data_signed, rng=42, mode='pepper', amount=0.15)
    proportion = float((cam_noisy_signed == -1).sum() - orig_zeros) / (cam.shape[0] * cam.shape[1])
    assert abs(amount - proportion) <= tolerance