import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_mixins_have_slots(self):
    mixin = NDArrayOperatorsMixin()
    assert_raises(AttributeError, mixin.__setattr__, 'not_a_real_attr', 1)
    m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
    wm = WrappedArray(m)
    assert_raises(AttributeError, wm.__setattr__, 'not_an_attr', 2)