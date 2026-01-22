import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_view_simple_dtype(self):
    mrec, a, b, arr = self.data
    ntype = (float, 2)
    test = mrec.view(ntype)
    assert_(isinstance(test, ma.MaskedArray))
    assert_equal(test, np.array(list(zip(a, b)), dtype=float))
    assert_(test[3, 1] is ma.masked)