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
def test_label_sequence():
    a = np.empty((2, 2), dtype=int)
    a[:, :] = 2
    ps = regionprops(a)
    assert len(ps) == 1
    assert ps[0].label == 2