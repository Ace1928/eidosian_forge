import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('dtype', ['str', 'object'])
def test_label_encoder_str_bad_shape(dtype):
    le = LabelEncoder()
    le.fit(np.array(['apple', 'orange'], dtype=dtype))
    msg = 'should be a 1d array'
    with pytest.raises(ValueError, match=msg):
        le.transform('apple')