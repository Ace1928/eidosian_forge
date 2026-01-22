import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_allany_onmatrices(self):
    x = np.array([[0.13, 0.26, 0.9], [0.28, 0.33, 0.63], [0.31, 0.87, 0.7]])
    X = np.matrix(x)
    m = np.array([[True, False, False], [False, False, False], [True, True, False]], dtype=np.bool_)
    mX = masked_array(X, mask=m)
    mXbig = mX > 0.5
    mXsmall = mX < 0.5
    assert_(not mXbig.all())
    assert_(mXbig.any())
    assert_equal(mXbig.all(0), np.matrix([False, False, True]))
    assert_equal(mXbig.all(1), np.matrix([False, False, True]).T)
    assert_equal(mXbig.any(0), np.matrix([False, False, True]))
    assert_equal(mXbig.any(1), np.matrix([True, True, True]).T)
    assert_(not mXsmall.all())
    assert_(mXsmall.any())
    assert_equal(mXsmall.all(0), np.matrix([True, True, False]))
    assert_equal(mXsmall.all(1), np.matrix([False, False, False]).T)
    assert_equal(mXsmall.any(0), np.matrix([True, True, False]))
    assert_equal(mXsmall.any(1), np.matrix([True, True, False]).T)