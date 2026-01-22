import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_masked_binary_operations(self):
    x, mx = self.data
    assert_(isinstance(add(mx, mx), MMatrix))
    assert_(isinstance(add(mx, x), MMatrix))
    assert_equal(add(mx, x), mx + x)
    assert_(isinstance(add(mx, mx)._data, np.matrix))
    with assert_warns(DeprecationWarning):
        assert_(isinstance(add.outer(mx, mx), MMatrix))
    assert_(isinstance(hypot(mx, mx), MMatrix))
    assert_(isinstance(hypot(mx, x), MMatrix))