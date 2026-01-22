import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
@pytest.mark.parametrize('spacing', [[2.1, 2.2], [2.0, 2.0], [2, 2]])
def test_spacing_parameter_2d(spacing):
    """Test the _normalize_spacing code."""
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))