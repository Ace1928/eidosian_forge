import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('params, expected_dtype', [({'arrays': (np.array([], dtype=np.int64), np.array([], dtype=np.int64)), 'check_contents': True}, np.int32), ({'arrays': np.array([1], dtype=np.int64), 'check_contents': True}, np.int32), ({'arrays': np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32), 'check_contents': True}, np.int64), ({'arrays': np.array([1], dtype=np.int32), 'check_contents': True, 'maxval': np.iinfo(np.int32).max + 1}, np.int64), ({'arrays': np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32), 'check_contents': True, 'maxval': 1}, np.int64)])
def test_smallest_admissible_index_dtype_by_checking_contents(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` using the dtype of the
    arrays but as well the contents.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype