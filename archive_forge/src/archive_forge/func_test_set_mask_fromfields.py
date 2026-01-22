import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_mask_fromfields(self):
    mbase = self.base.copy().view(mrecarray)
    nmask = np.array([(0, 1, 0), (0, 1, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)], dtype=[('a', bool), ('b', bool), ('c', bool)])
    mbase.mask = nmask
    assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
    assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
    assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
    mbase.mask = False
    mbase.fieldmask = nmask
    assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
    assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
    assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])