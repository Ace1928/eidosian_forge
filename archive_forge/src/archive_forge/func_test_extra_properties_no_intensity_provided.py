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
def test_extra_properties_no_intensity_provided():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(intensity_median,))[0]
        _ = region.intensity_median