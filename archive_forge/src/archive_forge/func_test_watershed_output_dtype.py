import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64])
def test_watershed_output_dtype(dtype):
    image = np.zeros((100, 100))
    markers = np.zeros((100, 100), dtype)
    out = watershed(image, markers)
    assert out.dtype == markers.dtype