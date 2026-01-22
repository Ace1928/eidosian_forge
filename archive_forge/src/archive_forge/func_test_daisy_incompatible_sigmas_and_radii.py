import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_daisy_incompatible_sigmas_and_radii():
    img = img_as_float(data.astronaut()[:64, :64].mean(axis=2))
    sigmas = [1, 2]
    radii = [1, 2]
    with pytest.raises(ValueError):
        daisy(img, sigmas=sigmas, ring_radii=radii)