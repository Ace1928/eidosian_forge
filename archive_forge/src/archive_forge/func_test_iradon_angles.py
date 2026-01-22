import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def test_iradon_angles():
    """
    Test with different number of projections
    """
    size = 100
    image = np.tri(size) + np.tri(size)[::-1]
    nb_angles = 200
    theta = np.linspace(0, 180, nb_angles, endpoint=False)
    radon_image_200 = radon(image, theta=theta, circle=False)
    reconstructed = iradon(radon_image_200, circle=False)
    delta_200 = np.mean(abs(_rescale_intensity(image) - _rescale_intensity(reconstructed)))
    assert delta_200 < 0.03
    nb_angles = 80
    radon_image_80 = radon(image, theta=theta, circle=False)
    s = radon_image_80.sum(axis=0)
    assert np.allclose(s, s[0], rtol=0.01)
    reconstructed = iradon(radon_image_80, circle=False)
    delta_80 = np.mean(abs(image / np.max(image) - reconstructed / np.max(reconstructed)))
    assert delta_80 > delta_200