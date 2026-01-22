import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_mask(self):
    base = self.base.copy()
    mbase = base.view(mrecarray)
    mbase.mask = masked
    assert_equal(ma.getmaskarray(mbase['b']), [1] * 5)
    assert_equal(mbase['a']._mask, mbase['b']._mask)
    assert_equal(mbase['a']._mask, mbase['c']._mask)
    assert_equal(mbase._mask.tolist(), np.array([(1, 1, 1)] * 5, dtype=bool))
    mbase.mask = nomask
    assert_equal(ma.getmaskarray(mbase['c']), [0] * 5)
    assert_equal(mbase._mask.tolist(), np.array([(0, 0, 0)] * 5, dtype=bool))