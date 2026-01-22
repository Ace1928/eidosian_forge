import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
def match_score(image, frequency):
    gabor_responses = gabor(image, frequency)
    return np.mean(np.hypot(*gabor_responses))