import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
def test_check_unknown_with_both_missing_values():
    values = np.array([np.nan, 'a', 'c', 'c', None, np.nan, None], dtype=object)
    diff = _check_unknown(values, known_values=np.array(['a', 'c'], dtype=object))
    assert diff[0] is None
    assert np.isnan(diff[1])
    diff, valid_mask = _check_unknown(values, known_values=np.array(['a', 'c'], dtype=object), return_mask=True)
    assert diff[0] is None
    assert np.isnan(diff[1])
    assert_array_equal(valid_mask, [False, True, True, True, False, False, False])