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
def test_extra_properties_mixed():
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, extra_properties=(intensity_median, pixelcount))[0]
    assert region.intensity_median == np.median(INTENSITY_SAMPLE[SAMPLE == 1])
    assert region.pixelcount == np.sum(SAMPLE == 1)