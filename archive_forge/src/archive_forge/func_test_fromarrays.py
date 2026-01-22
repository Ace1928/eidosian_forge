import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_fromarrays(self):
    _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
    _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
    _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
    mrec, nrec, _ = self.data
    for f, l in zip(('a', 'b', 'c'), (_a, _b, _c)):
        assert_equal(getattr(mrec, f)._mask, l._mask)
    _x = ma.array([1, 1.1, 'one'], mask=[1, 0, 0], dtype=object)
    assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])