import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_exotic_formats(self):
    easy = mrecarray(1, dtype=[('i', int), ('s', '|S8'), ('f', float)])
    easy[0] = masked
    assert_equal(easy.filled(1).item(), (1, b'1', 1.0))
    solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
    solo[0] = masked
    assert_equal(solo.filled(1).item(), np.array((1,), dtype=solo.dtype).item())
    mult = mrecarray(2, dtype='i4, (2,3)float, float')
    mult[0] = masked
    mult[1] = (1, 1, 1)
    mult.filled(0)
    assert_equal_records(mult.filled(0), np.array([(0, 0, 0), (1, 1, 1)], dtype=mult.dtype))