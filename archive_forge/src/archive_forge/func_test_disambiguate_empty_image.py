import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import fourier_shift
import scipy.fft as fft
from skimage import img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_stacklevel
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, binary_blobs, eagle
from skimage.registration._phase_cross_correlation import (
@pytest.mark.parametrize('null_images', [(1, 0), (0, 1), (0, 0)])
def test_disambiguate_empty_image(null_images):
    """When the image is empty, disambiguation becomes degenerate."""
    image = camera()
    with pytest.warns(UserWarning) as records:
        shift, error, phasediff = phase_cross_correlation(image * null_images[0], image * null_images[1], disambiguate=True)
        assert_stacklevel(records, offset=-3)
    np.testing.assert_array_equal(shift, np.array([0.0, 0.0]))
    assert np.isnan(error)
    assert phasediff == 0.0
    assert len(records) == 2
    assert 'Could not determine real-space shift' in records[0].message.args[0]
    assert 'Could not determine RMS error between images' in records[1].message.args[0]