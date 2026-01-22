import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_dtype_creation(self):
    for unit in ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'μs', 'ns', 'ps', 'fs', 'as']:
        dt1 = np.dtype('M8[750%s]' % unit)
        assert_(dt1 == np.dtype('datetime64[750%s]' % unit))
        dt2 = np.dtype('m8[%s]' % unit)
        assert_(dt2 == np.dtype('timedelta64[%s]' % unit))
    assert_equal(str(np.dtype('M8')), 'datetime64')
    assert_equal(np.dtype('=M8'), np.dtype('M8'))
    assert_equal(np.dtype('=M8[s]'), np.dtype('M8[s]'))
    assert_(np.dtype('>M8') == np.dtype('M8') or np.dtype('<M8') == np.dtype('M8'))
    assert_(np.dtype('>M8[D]') == np.dtype('M8[D]') or np.dtype('<M8[D]') == np.dtype('M8[D]'))
    assert_(np.dtype('>M8') != np.dtype('<M8'))
    assert_equal(np.dtype('=m8'), np.dtype('m8'))
    assert_equal(np.dtype('=m8[s]'), np.dtype('m8[s]'))
    assert_(np.dtype('>m8') == np.dtype('m8') or np.dtype('<m8') == np.dtype('m8'))
    assert_(np.dtype('>m8[D]') == np.dtype('m8[D]') or np.dtype('<m8[D]') == np.dtype('m8[D]'))
    assert_(np.dtype('>m8') != np.dtype('<m8'))
    assert_raises(TypeError, np.dtype, 'M8[badunit]')
    assert_raises(TypeError, np.dtype, 'm8[badunit]')
    assert_raises(TypeError, np.dtype, 'M8[YY]')
    assert_raises(TypeError, np.dtype, 'm8[YY]')
    assert_raises(TypeError, np.dtype, 'm4')
    assert_raises(TypeError, np.dtype, 'M7')
    assert_raises(TypeError, np.dtype, 'm7')
    assert_raises(TypeError, np.dtype, 'M16')
    assert_raises(TypeError, np.dtype, 'm16')
    assert_raises(TypeError, np.dtype, 'M8[3000000000ps]')