import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
@pytest.mark.parametrize('values, uniques, expected_counts', [(np.array([1] * 10 + [2] * 4 + [3] * 15), np.array([1, 2, 3]), [10, 4, 15]), (np.array([1] * 10 + [2] * 4 + [3] * 15), np.array([1, 2, 3, 5]), [10, 4, 15, 0]), (np.array([np.nan] * 10 + [2] * 4 + [3] * 15), np.array([2, 3, np.nan]), [4, 15, 10]), (np.array(['b'] * 4 + ['a'] * 16 + ['c'] * 20, dtype=object), ['a', 'b', 'c'], [16, 4, 20]), (np.array(['b'] * 4 + ['a'] * 16 + ['c'] * 20, dtype=object), ['c', 'b', 'a'], [20, 4, 16]), (np.array([np.nan] * 4 + ['a'] * 16 + ['c'] * 20, dtype=object), ['c', np.nan, 'a'], [20, 4, 16]), (np.array(['b'] * 4 + ['a'] * 16 + ['c'] * 20, dtype=object), ['a', 'b', 'c', 'e'], [16, 4, 20, 0])])
def test_get_counts(values, uniques, expected_counts):
    counts = _get_counts(values, uniques)
    assert_array_equal(counts, expected_counts)