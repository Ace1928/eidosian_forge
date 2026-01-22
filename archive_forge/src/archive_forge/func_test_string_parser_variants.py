import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_string_parser_variants(self):
    assert_equal(np.array(['1980-02-29T01:02:03'], np.dtype('M8[s]')), np.array(['1980-02-29 01:02:03'], np.dtype('M8[s]')))
    assert_equal(np.array(['+1980-02-29T01:02:03'], np.dtype('M8[s]')), np.array(['+1980-02-29 01:02:03'], np.dtype('M8[s]')))
    assert_equal(np.array(['-1980-02-29T01:02:03'], np.dtype('M8[s]')), np.array(['-1980-02-29 01:02:03'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.array(['+1980-02-29T01:02:03'], np.dtype('M8[s]')), np.array(['+1980-02-29 01:02:03Z'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.array(['-1980-02-29T01:02:03'], np.dtype('M8[s]')), np.array(['-1980-02-29 01:02:03Z'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.array(['1980-02-29T02:02:03'], np.dtype('M8[s]')), np.array(['1980-02-29 00:32:03-0130'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.array(['1980-02-28T22:32:03'], np.dtype('M8[s]')), np.array(['1980-02-29 00:02:03+01:30'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.array(['1980-02-29T02:32:03.506'], np.dtype('M8[s]')), np.array(['1980-02-29 00:32:03.506-02'], np.dtype('M8[s]')))
    with assert_warns(DeprecationWarning):
        assert_equal(np.datetime64('1977-03-02T12:30-0230'), np.datetime64('1977-03-02T15:00'))