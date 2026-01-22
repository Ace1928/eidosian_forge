import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def check_sinogram_circle_to_square(size):
    from skimage.transform.radon_transform import _sinogram_circle_to_square
    image = _random_circle((size, size))
    theta = np.linspace(0.0, 180.0, size, False)
    sinogram_circle = radon(image, theta, circle=True)

    def argmax_shape(a):
        return np.unravel_index(np.argmax(a), a.shape)
    print('\n\targmax of circle:', argmax_shape(sinogram_circle))
    sinogram_square = radon(image, theta, circle=False)
    print('\targmax of square:', argmax_shape(sinogram_square))
    sinogram_circle_to_square = _sinogram_circle_to_square(sinogram_circle)
    print('\targmax of circle to square:', argmax_shape(sinogram_circle_to_square))
    error = abs(sinogram_square - sinogram_circle_to_square)
    print(np.mean(error), np.max(error))
    assert argmax_shape(sinogram_square) == argmax_shape(sinogram_circle_to_square)