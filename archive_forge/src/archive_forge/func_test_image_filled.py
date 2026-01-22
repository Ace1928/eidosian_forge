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
def test_image_filled():
    img = regionprops(SAMPLE)[0].image_filled
    assert_array_equal(img, SAMPLE)
    img = regionprops(SAMPLE, spacing=(1, 4))[0].image_filled
    assert_array_equal(img, SAMPLE)