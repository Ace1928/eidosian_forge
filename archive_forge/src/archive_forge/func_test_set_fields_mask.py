import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_fields_mask(self):
    base = self.base.copy()
    mbase = base.view(mrecarray)
    mbase['a'][-2] = masked
    assert_equal(mbase.a, [1, 2, 3, 4, 5])
    assert_equal(mbase.a._mask, [0, 1, 0, 1, 1])
    mbase = fromarrays([np.arange(5), np.random.rand(5)], dtype=[('a', int), ('b', float)])
    mbase['a'][-2] = masked
    assert_equal(mbase.a, [0, 1, 2, 3, 4])
    assert_equal(mbase.a._mask, [0, 0, 0, 1, 0])