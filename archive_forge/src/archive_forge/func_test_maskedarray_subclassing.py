import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_maskedarray_subclassing(self):
    x, mx = self.data
    assert_(isinstance(mx._data, np.matrix))