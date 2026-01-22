import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_flat(self):
    test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
    assert_equal(test.flat[1], 2)
    assert_equal(test.flat[2], masked)
    assert_(np.all(test.flat[0:2] == test[0, 0:2]))
    test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
    test.flat = masked_array([3, 2, 1], mask=[1, 0, 0])
    control = masked_array(np.matrix([[3, 2, 1]]), mask=[1, 0, 0])
    assert_equal(test, control)
    test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
    testflat = test.flat
    testflat[:] = testflat[[2, 1, 0]]
    assert_equal(test, control)
    testflat[0] = 9
    a = masked_array(np.matrix(np.eye(2)), mask=0)
    b = a.flat
    b01 = b[:2]
    assert_equal(b01.data, np.array([[1.0, 0.0]]))
    assert_equal(b01.mask, np.array([[False, False]]))