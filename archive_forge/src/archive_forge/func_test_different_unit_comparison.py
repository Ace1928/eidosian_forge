import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_different_unit_comparison(self):
    for unit1 in ['Y', 'M', 'D']:
        dt1 = np.dtype('M8[%s]' % unit1)
        for unit2 in ['Y', 'M', 'D']:
            dt2 = np.dtype('M8[%s]' % unit2)
            assert_equal(np.array('1945', dtype=dt1), np.array('1945', dtype=dt2))
            assert_equal(np.array('1970', dtype=dt1), np.array('1970', dtype=dt2))
            assert_equal(np.array('9999', dtype=dt1), np.array('9999', dtype=dt2))
            assert_equal(np.array('10000', dtype=dt1), np.array('10000-01-01', dtype=dt2))
            assert_equal(np.datetime64('1945', unit1), np.datetime64('1945', unit2))
            assert_equal(np.datetime64('1970', unit1), np.datetime64('1970', unit2))
            assert_equal(np.datetime64('9999', unit1), np.datetime64('9999', unit2))
            assert_equal(np.datetime64('10000', unit1), np.datetime64('10000-01-01', unit2))
    for unit1 in ['6h', 'h', 'm', 's', '10ms', 'ms', 'us']:
        dt1 = np.dtype('M8[%s]' % unit1)
        for unit2 in ['h', 'm', 's', 'ms', 'us']:
            dt2 = np.dtype('M8[%s]' % unit2)
            assert_equal(np.array('1945-03-12T18', dtype=dt1), np.array('1945-03-12T18', dtype=dt2))
            assert_equal(np.array('1970-03-12T18', dtype=dt1), np.array('1970-03-12T18', dtype=dt2))
            assert_equal(np.array('9999-03-12T18', dtype=dt1), np.array('9999-03-12T18', dtype=dt2))
            assert_equal(np.array('10000-01-01T00', dtype=dt1), np.array('10000-01-01T00', dtype=dt2))
            assert_equal(np.datetime64('1945-03-12T18', unit1), np.datetime64('1945-03-12T18', unit2))
            assert_equal(np.datetime64('1970-03-12T18', unit1), np.datetime64('1970-03-12T18', unit2))
            assert_equal(np.datetime64('9999-03-12T18', unit1), np.datetime64('9999-03-12T18', unit2))
            assert_equal(np.datetime64('10000-01-01T00', unit1), np.datetime64('10000-01-01T00', unit2))
    for unit1 in ['D', '12h', 'h', 'm', 's', '4s', 'ms', 'us']:
        dt1 = np.dtype('M8[%s]' % unit1)
        for unit2 in ['D', 'h', 'm', 's', 'ms', 'us']:
            dt2 = np.dtype('M8[%s]' % unit2)
            assert_(np.equal(np.array('1932-02-17', dtype='M').astype(dt1), np.array('1932-02-17T00:00:00', dtype='M').astype(dt2), casting='unsafe'))
            assert_(np.equal(np.array('10000-04-27', dtype='M').astype(dt1), np.array('10000-04-27T00:00:00', dtype='M').astype(dt2), casting='unsafe'))
    a = np.array('2012-12-21', dtype='M8[D]')
    b = np.array(3, dtype='m8[D]')
    assert_raises(TypeError, np.less, a, b)
    assert_raises(TypeError, np.less, a, b, casting='unsafe')