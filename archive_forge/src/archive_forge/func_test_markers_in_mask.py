import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_markers_in_mask():
    data = blob
    mask = data != 255
    out = watershed(data, 25, connectivity=2, mask=mask)
    assert np.all(out[~mask] == 0)