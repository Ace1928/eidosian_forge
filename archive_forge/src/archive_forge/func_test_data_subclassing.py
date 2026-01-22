import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_data_subclassing(self):
    x = np.arange(5)
    m = [0, 0, 1, 0, 0]
    xsub = SubArray(x)
    xmsub = masked_array(xsub, mask=m)
    assert_(isinstance(xmsub, MaskedArray))
    assert_equal(xmsub._data, xsub)
    assert_(isinstance(xmsub._data, SubArray))