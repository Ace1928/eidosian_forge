import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_rsh():
    np.random.seed(132345)
    x = np.random.randn(100)
    res = ms.rsh(x)
    assert_(res.shape == x.shape)
    res = ms.rsh(x, points=[0, 1.0])
    assert_(res.size == 2)