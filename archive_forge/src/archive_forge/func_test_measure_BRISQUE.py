import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_BRISQUE():
    vidpaths = skvideo.datasets.bigbuckbunny()
    dis = skvideo.io.vread(vidpaths, as_grey=True)
    dis = dis[0, :200, :200]
    features = skvideo.measure.brisque_features(dis)
    output = np.array([2.2890000343, 0.2322334051, 0.8130000234, 0.071422264, 0.0303122569, 0.0790375844, 0.7820000052, 0.125390932, 0.0196695272, 0.1092280298, 0.800999999, 0.0333177634, 0.0419092514, 0.0649642125, 0.800999999, 0.0416957438, 0.0396158583, 0.068546854, 3.1700000763, 0.3377875388, 0.9840000272, 0.0400288589, 0.0888380781, 0.125952065, 0.9520000219, 0.1778325588, 0.0371656679, 0.2002325803, 0.8489999771, -0.0157390144, 0.1383629888, 0.1215882078, 0.8629999757, 0.0312444586, 0.1079876497, 0.1403252929])
    for i in range(features.shape[1]):
        assert_almost_equal(features[0, i], output[i], decimal=10)