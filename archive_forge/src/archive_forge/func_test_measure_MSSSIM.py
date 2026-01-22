import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_MSSSIM():
    vidpaths = skvideo.datasets.fullreferencepair()
    ref = skvideo.io.vread(vidpaths[0], as_grey=True)
    dis = skvideo.io.vread(vidpaths[1], as_grey=True)
    ref = np.pad(ref, [(0, 0), (20, 20), (20, 20), (0, 0)], mode='constant')
    dis = np.pad(dis, [(0, 0), (20, 20), (20, 20), (0, 0)], mode='constant')
    scores = skvideo.measure.msssim(ref, dis)
    assert_almost_equal(np.mean(scores), 0.909682154655457, decimal=15)