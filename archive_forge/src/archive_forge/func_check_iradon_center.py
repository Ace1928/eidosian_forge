import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def check_iradon_center(size, theta, circle):
    debug = False
    if circle:
        sinogram = np.zeros((size, 1), dtype=float)
        sinogram[size // 2, 0] = 1.0
    else:
        diagonal = int(np.ceil(np.sqrt(2) * size))
        sinogram = np.zeros((diagonal, 1), dtype=float)
        sinogram[sinogram.shape[0] // 2, 0] = 1.0
    maxpoint = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    print('shape of generated sinogram', sinogram.shape)
    print('maximum in generated sinogram', maxpoint)
    reconstruction = iradon(sinogram, theta=[theta], circle=circle)
    reconstruction_opposite = iradon(sinogram, theta=[theta + 180], circle=circle)
    print('rms deviance:', np.sqrt(np.mean((reconstruction_opposite - reconstruction) ** 2)))
    if debug and has_mpl:
        import matplotlib.pyplot as plt
        imkwargs = dict(cmap='gray', interpolation='nearest')
        plt.figure()
        plt.subplot(221)
        plt.imshow(sinogram, **imkwargs)
        plt.subplot(222)
        plt.imshow(reconstruction_opposite - reconstruction, **imkwargs)
        plt.subplot(223)
        plt.imshow(reconstruction, **imkwargs)
        plt.subplot(224)
        plt.imshow(reconstruction_opposite, **imkwargs)
        plt.show()
    assert np.allclose(reconstruction, reconstruction_opposite)