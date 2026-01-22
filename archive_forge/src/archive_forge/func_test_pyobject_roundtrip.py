import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_pyobject_roundtrip(self):
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1020040340, -2942398, -1, 0, 1, 234523453, 1199164176], dtype=np.int64)
    for unit in ['M8[D]', 'M8[W]', 'M8[M]', 'M8[Y]']:
        b = a.copy().view(dtype=unit)
        b[0] = '-0001-01-01'
        b[1] = '-0001-12-31'
        b[2] = '0000-01-01'
        b[3] = '0001-01-01'
        b[4] = '1969-12-31'
        b[5] = '1970-01-01'
        b[6] = '9999-12-31'
        b[7] = '10000-01-01'
        b[8] = 'NaT'
        assert_equal(b.astype(object).astype(unit), b, 'Error roundtripping unit %s' % unit)
    for unit in ['M8[as]', 'M8[16fs]', 'M8[ps]', 'M8[us]', 'M8[300as]', 'M8[20us]']:
        b = a.copy().view(dtype=unit)
        b[0] = '-0001-01-01T00'
        b[1] = '-0001-12-31T00'
        b[2] = '0000-01-01T00'
        b[3] = '0001-01-01T00'
        b[4] = '1969-12-31T23:59:59.999999'
        b[5] = '1970-01-01T00'
        b[6] = '9999-12-31T23:59:59.999999'
        b[7] = '10000-01-01T00'
        b[8] = 'NaT'
        assert_equal(b.astype(object).astype(unit), b, 'Error roundtripping unit %s' % unit)