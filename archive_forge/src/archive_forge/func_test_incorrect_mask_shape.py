import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_incorrect_mask_shape():
    image = np.ones((5, 6))
    mask = np.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers=4, mask=mask)