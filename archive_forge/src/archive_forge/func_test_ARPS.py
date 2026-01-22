from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.motion
import skvideo.datasets
def test_ARPS():
    dat = getmockdata()
    mvec = skvideo.motion.blockMotion(dat, 'ARPS')
    mvec = mvec.astype(np.float32)
    xmean = np.mean(mvec[:, :, :, 0])
    ymean = np.mean(mvec[:, :, :, 1])
    assert_almost_equal(ymean, 0.013888888992369, decimal=15)
    assert_almost_equal(xmean, -0.333333343267441, decimal=15)