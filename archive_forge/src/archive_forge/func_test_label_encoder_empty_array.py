import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('values', [np.array([2, 1, 3, 1, 3], dtype='int64'), np.array(['b', 'a', 'c', 'a', 'c'], dtype=object), np.array(['b', 'a', 'c', 'a', 'c'])], ids=['int64', 'object', 'str'])
def test_label_encoder_empty_array(values):
    le = LabelEncoder()
    le.fit(values)
    transformed = le.transform([])
    assert_array_equal(np.array([]), transformed)
    inverse_transformed = le.inverse_transform([])
    assert_array_equal(np.array([]), inverse_transformed)