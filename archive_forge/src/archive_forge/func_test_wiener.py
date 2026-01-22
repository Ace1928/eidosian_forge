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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_wiener(dtype, ndim):
    """
    currently only performs pixelwise comparison to
    precomputed result in 2d case.
    """
    rng = np.random.RandomState(0)
    psf = np.ones([5] * ndim, dtype=dtype) / 5 ** ndim
    if ndim != 2:
        test_img = rng.randint(0, 100, [50] * ndim)
    else:
        test_img = util.img_as_float(camera())
    data = convolve(test_img, psf, 'same')
    data += 0.1 * data.std() * rng.standard_normal(data.shape)
    data = data.astype(dtype, copy=False)
    deconvolved = restoration.wiener(data, psf, 0.05)
    assert deconvolved.dtype == _supported_float_type(dtype)
    if ndim == 2:
        rtol, atol = _get_rtol_atol(dtype)
        path = fetch('restoration/tests/camera_wiener.npy')
        np.testing.assert_allclose(deconvolved, np.load(path), rtol=rtol, atol=atol)
    _, laplacian = uft.laplacian(ndim, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    assert otf.real.dtype == _supported_float_type(dtype)
    deconvolved = restoration.wiener(data, otf, 0.05, reg=laplacian, is_real=False)
    assert deconvolved.real.dtype == _supported_float_type(dtype)
    if ndim == 2:
        np.testing.assert_allclose(np.real(deconvolved), np.load(path), rtol=rtol, atol=atol)