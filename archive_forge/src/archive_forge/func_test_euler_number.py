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
def test_euler_number():
    for spacing in [(1, 1), (2.1, 0.9)]:
        en = regionprops(SAMPLE, spacing=spacing)[0].euler_number
        assert en == 0
        SAMPLE_mod = SAMPLE.copy()
        SAMPLE_mod[7, -3] = 0
        en = regionprops(SAMPLE_mod, spacing=spacing)[0].euler_number
        assert en == -1
        en = euler_number(SAMPLE, 1)
        assert en == 2
        en = euler_number(SAMPLE_mod, 1)
        assert en == 1
    en = euler_number(SAMPLE_3D, 1)
    assert en == 1
    en = euler_number(SAMPLE_3D, 3)
    assert en == 1
    SAMPLE_3D_2 = np.zeros((100, 100, 100))
    SAMPLE_3D_2[40:60, 40:60, 40:60] = 1
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 1
    SAMPLE_3D_2[45:55, 45:55, 45:55] = 0
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 2