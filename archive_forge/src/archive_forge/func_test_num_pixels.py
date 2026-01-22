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
def test_num_pixels():
    num_pixels = regionprops(SAMPLE)[0].num_pixels
    assert num_pixels == 72
    num_pixels = regionprops(SAMPLE, spacing=(2, 1))[0].num_pixels
    assert num_pixels == 72