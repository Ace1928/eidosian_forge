import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_nmi_different_sizes():
    assert normalized_mutual_information(cam[:, :400], cam[:400, :]) > 1