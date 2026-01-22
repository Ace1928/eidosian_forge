from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_dictionary_to_numpy():
    expected = pa.array(['foo', 'bar', None, 'foo']).to_numpy(zero_copy_only=False)
    a = pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 0]), pa.array(['foo', 'bar']))
    np.testing.assert_array_equal(a.to_numpy(zero_copy_only=False), expected)
    with pytest.raises(pa.ArrowInvalid):
        a.to_numpy(zero_copy_only=True)
    anonulls = pa.DictionaryArray.from_arrays(pa.array([0, 1, 1, 0]), pa.array(['foo', 'bar']))
    expected = pa.array(['foo', 'bar', 'bar', 'foo']).to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(anonulls.to_numpy(zero_copy_only=False), expected)
    with pytest.raises(pa.ArrowInvalid):
        anonulls.to_numpy(zero_copy_only=True)
    afloat = pa.DictionaryArray.from_arrays(pa.array([0, 1, 1, 0]), pa.array([13.7, 11.0]))
    expected = pa.array([13.7, 11.0, 11.0, 13.7]).to_numpy()
    np.testing.assert_array_equal(afloat.to_numpy(zero_copy_only=True), expected)
    np.testing.assert_array_equal(afloat.to_numpy(zero_copy_only=False), expected)
    afloat2 = pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 0]), pa.array([13.7, 11.0]))
    expected = pa.array([13.7, 11.0, None, 13.7]).to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(afloat2.to_numpy(zero_copy_only=False), expected, equal_nan=True)
    aints = pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 0]), pa.array([7, 11]))
    expected = pa.array([7, 11, None, 7]).to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(aints.to_numpy(zero_copy_only=False), expected, equal_nan=True)