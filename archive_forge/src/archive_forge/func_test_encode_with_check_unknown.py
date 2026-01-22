import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
def test_encode_with_check_unknown():
    uniques = np.array([1, 2, 3])
    values = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match='y contains previously unseen labels'):
        _encode(values, uniques=uniques, check_unknown=True)
    _encode(values, uniques=uniques, check_unknown=False)
    uniques = np.array(['a', 'b', 'c'], dtype=object)
    values = np.array(['a', 'b', 'c', 'd'], dtype=object)
    with pytest.raises(ValueError, match='y contains previously unseen labels'):
        _encode(values, uniques=uniques, check_unknown=False)