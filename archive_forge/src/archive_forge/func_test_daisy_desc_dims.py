import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_daisy_desc_dims():
    img = img_as_float(data.astronaut()[:128, :128].mean(axis=2))
    rings = 2
    histograms = 4
    orientations = 3
    descs = daisy(img, rings=rings, histograms=histograms, orientations=orientations)
    assert descs.shape[2] == (rings * histograms + 1) * orientations
    rings = 4
    histograms = 5
    orientations = 13
    descs = daisy(img, rings=rings, histograms=histograms, orientations=orientations)
    assert descs.shape[2] == (rings * histograms + 1) * orientations