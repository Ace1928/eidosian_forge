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
@pytest.mark.parametrize('spacing', [(1, 1j), 1 + 0j])
def test_spacing_parameter_complex_input(spacing):
    """Test the _normalize_spacing code."""
    with pytest.raises(TypeError, match="Element of spacing isn't float or integer type, got"):
        regionprops(SAMPLE, spacing=spacing)[0].centroid