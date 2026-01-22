import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_float_special(self):
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for inf in [np.inf, -np.inf]:
            a = np.array([[inf, np.nan], [np.nan, np.nan]])
            assert_equal(np.nanmedian(a, axis=0), [inf, np.nan])
            assert_equal(np.nanmedian(a, axis=1), [inf, np.nan])
            assert_equal(np.nanmedian(a), inf)
            a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
            assert_equal(np.nanmedian(a), inf)
            assert_equal(np.nanmedian(a, axis=0), [np.nan, np.nan, inf])
            assert_equal(np.nanmedian(a, axis=1), inf)
            a = np.array([[inf, inf], [inf, inf]])
            assert_equal(np.nanmedian(a, axis=1), inf)
            a = np.array([[inf, 7, -inf, -9], [-10, np.nan, np.nan, 5], [4, np.nan, np.nan, inf]], dtype=np.float32)
            if inf > 0:
                assert_equal(np.nanmedian(a, axis=0), [4.0, 7.0, -inf, 5.0])
                assert_equal(np.nanmedian(a), 4.5)
            else:
                assert_equal(np.nanmedian(a, axis=0), [-10.0, 7.0, -inf, -9.0])
                assert_equal(np.nanmedian(a), -2.5)
            assert_equal(np.nanmedian(a, axis=-1), [-1.0, -2.5, inf])
            for i in range(0, 10):
                for j in range(1, 10):
                    a = np.array([[np.nan] * i + [inf] * j] * 2)
                    assert_equal(np.nanmedian(a), inf)
                    assert_equal(np.nanmedian(a, axis=1), inf)
                    assert_equal(np.nanmedian(a, axis=0), [np.nan] * i + [inf] * j)
                    a = np.array([[np.nan] * i + [-inf] * j] * 2)
                    assert_equal(np.nanmedian(a), -inf)
                    assert_equal(np.nanmedian(a, axis=1), -inf)
                    assert_equal(np.nanmedian(a, axis=0), [np.nan] * i + [-inf] * j)