import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_view_flexible_type(self):
    mrec, a, b, arr = self.data
    alttype = [('A', float), ('B', float)]
    test = mrec.view(alttype)
    assert_(isinstance(test, MaskedRecords))
    assert_equal_records(test, arr.view(alttype))
    assert_(test['B'][3] is masked)
    assert_equal(test.dtype, np.dtype(alttype))
    assert_(test._fill_value is None)