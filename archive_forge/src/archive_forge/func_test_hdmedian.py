import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_hdmedian():
    x = ma.arange(11)
    assert_allclose(ms.hdmedian(x), 5, rtol=1e-14)
    x.mask = ma.make_mask(x)
    x.mask[:7] = False
    assert_allclose(ms.hdmedian(x), 3, rtol=1e-14)
    assert_(ms.hdmedian(x, var=True).size == 2)
    x2 = ma.arange(22).reshape((11, 2))
    assert_allclose(ms.hdmedian(x2, axis=0), [10, 11])
    x2.mask = ma.make_mask(x2)
    x2.mask[:7, :] = False
    assert_allclose(ms.hdmedian(x2, axis=0), [6, 7])