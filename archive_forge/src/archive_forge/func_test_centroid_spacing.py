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
def test_centroid_spacing():
    spacing = (1.8, 0.8)
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cY, cX))