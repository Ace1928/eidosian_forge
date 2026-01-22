import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_nmi():
    assert_almost_equal(normalized_mutual_information(cam, cam), 2)
    assert normalized_mutual_information(cam, cam_noisy) < normalized_mutual_information(cam, cam)