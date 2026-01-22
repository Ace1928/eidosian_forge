import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_pickling_subbaseclass(self):
    a = masked_array(np.matrix(list(range(10))), mask=[1, 0, 1, 0, 0] * 2)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled, a)
        assert_(isinstance(a_pickled._data, np.matrix))