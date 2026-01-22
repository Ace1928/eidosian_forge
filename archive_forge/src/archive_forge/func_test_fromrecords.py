import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_fromrecords(self):
    mrec, nrec, ddtype = self.data
    palist = [(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)]
    pa = recfromrecords(palist, names='c1, c2, c3, c4')
    mpa = fromrecords(palist, names='c1, c2, c3, c4')
    assert_equal_records(pa, mpa)
    _mrec = fromrecords(nrec)
    assert_equal(_mrec.dtype, mrec.dtype)
    for field in _mrec.dtype.names:
        assert_equal(getattr(_mrec, field), getattr(mrec._data, field))
    _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
    assert_equal(_mrec.dtype, [('c1', int), ('c2', float), ('c3', '|S5')])
    for f, n in zip(('c1', 'c2', 'c3'), ('a', 'b', 'c')):
        assert_equal(getattr(_mrec, f), getattr(mrec._data, n))
    _mrec = fromrecords(mrec)
    assert_equal(_mrec.dtype, mrec.dtype)
    assert_equal_records(_mrec._data, mrec.filled())
    assert_equal_records(_mrec._mask, mrec._mask)