import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_subtract(self):
    for dta, dtb, dtc, dtd, dte, dtnat, tda, tdb, tdc in [(np.array(['2012-12-21'], dtype='M8[D]'), np.array(['2012-12-24'], dtype='M8[D]'), np.array(['1940-12-24'], dtype='M8[D]'), np.array(['1940-12-24T00'], dtype='M8[h]'), np.array(['1940-12-23T13'], dtype='M8[h]'), np.array(['NaT'], dtype='M8[D]'), np.array([3], dtype='m8[D]'), np.array([11], dtype='m8[h]'), np.array([3 * 24 - 11], dtype='m8[h]')), (np.datetime64('2012-12-21', '[D]'), np.datetime64('2012-12-24', '[D]'), np.datetime64('1940-12-24', '[D]'), np.datetime64('1940-12-24T00', '[h]'), np.datetime64('1940-12-23T13', '[h]'), np.datetime64('NaT', '[D]'), np.timedelta64(3, '[D]'), np.timedelta64(11, '[h]'), np.timedelta64(3 * 24 - 11, '[h]'))]:
        assert_equal(tda - tdb, tdc)
        assert_equal((tda - tdb).dtype, np.dtype('m8[h]'))
        assert_equal(tdb - tda, -tdc)
        assert_equal((tdb - tda).dtype, np.dtype('m8[h]'))
        assert_equal(tdc - True, tdc - 1)
        assert_equal((tdc - True).dtype, np.dtype('m8[h]'))
        assert_equal(tdc - 3 * 24, -tdb)
        assert_equal((tdc - 3 * 24).dtype, np.dtype('m8[h]'))
        assert_equal(False - tdb, -tdb)
        assert_equal((False - tdb).dtype, np.dtype('m8[h]'))
        assert_equal(3 * 24 - tdb, tdc)
        assert_equal((3 * 24 - tdb).dtype, np.dtype('m8[h]'))
        assert_equal(dtb - True, dtb - 1)
        assert_equal(dtnat - True, dtnat)
        assert_equal((dtb - True).dtype, np.dtype('M8[D]'))
        assert_equal(dtb - 3, dta)
        assert_equal(dtnat - 3, dtnat)
        assert_equal((dtb - 3).dtype, np.dtype('M8[D]'))
        assert_equal(dtb - tda, dta)
        assert_equal(dtnat - tda, dtnat)
        assert_equal((dtb - tda).dtype, np.dtype('M8[D]'))
        assert_equal(np.subtract(dtc, tdb, casting='unsafe'), dte)
        assert_equal(np.subtract(dtc, tdb, casting='unsafe').dtype, np.dtype('M8[h]'))
        assert_equal(np.subtract(dtc, dtd, casting='unsafe'), np.timedelta64(0, 'h'))
        assert_equal(np.subtract(dtc, dtd, casting='unsafe').dtype, np.dtype('m8[h]'))
        assert_equal(np.subtract(dtd, dtc, casting='unsafe'), np.timedelta64(0, 'h'))
        assert_equal(np.subtract(dtd, dtc, casting='unsafe').dtype, np.dtype('m8[h]'))
        assert_raises(TypeError, np.subtract, tda, dta)
        assert_raises(TypeError, np.subtract, False, dta)
        assert_raises(TypeError, np.subtract, 3, dta)