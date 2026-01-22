import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_mask_fromarray(self):
    base = self.base.copy()
    mbase = base.view(mrecarray)
    mbase.mask = [1, 0, 0, 0, 1]
    assert_equal(mbase.a.mask, [1, 0, 0, 0, 1])
    assert_equal(mbase.b.mask, [1, 0, 0, 0, 1])
    assert_equal(mbase.c.mask, [1, 0, 0, 0, 1])
    mbase.mask = [0, 0, 0, 0, 1]
    assert_equal(mbase.a.mask, [0, 0, 0, 0, 1])
    assert_equal(mbase.b.mask, [0, 0, 0, 0, 1])
    assert_equal(mbase.c.mask, [0, 0, 0, 0, 1])