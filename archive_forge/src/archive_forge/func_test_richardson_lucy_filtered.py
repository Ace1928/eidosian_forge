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
@pytest.mark.parametrize('dtype_image', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('dtype_psf', [np.float32, np.float64])
def test_richardson_lucy_filtered(dtype_image, dtype_psf):
    if dtype_image == np.float64:
        atol = 1e-08
    else:
        atol = 1e-05
    test_img_astro = rgb2gray(astronaut())
    psf = np.ones((5, 5), dtype=dtype_psf) / 25
    data = convolve2d(test_img_astro, psf, 'same')
    data = data.astype(dtype_image, copy=False)
    deconvolved = restoration.richardson_lucy(data, psf, 5, filter_epsilon=1e-06)
    assert deconvolved.dtype == _supported_float_type(data.dtype)
    path = fetch('restoration/tests/astronaut_rl.npy')
    np.testing.assert_allclose(deconvolved, np.load(path), rtol=0.001, atol=atol)