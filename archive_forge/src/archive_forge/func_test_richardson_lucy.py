import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.signal import convolve2d, convolve
from skimage import restoration, util
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import astronaut, camera
from skimage.restoration import uft
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_richardson_lucy(ndim):
    psf = np.ones([5] * ndim, dtype=float) / 5 ** ndim
    if ndim != 2:
        test_img = np.random.randint(0, 100, [30] * ndim)
    else:
        test_img = util.img_as_float(camera())
    data = convolve(test_img, psf, 'same')
    rng = np.random.RandomState(0)
    data += 0.1 * data.std() * rng.standard_normal(data.shape)
    deconvolved = restoration.richardson_lucy(data, psf, num_iter=5)
    if ndim == 2:
        path = fetch('restoration/tests/camera_rl.npy')
        np.testing.assert_allclose(deconvolved, np.load(path), rtol=0.001)