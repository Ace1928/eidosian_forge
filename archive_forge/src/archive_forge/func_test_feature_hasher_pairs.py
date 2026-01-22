import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_feature_hasher_pairs():
    raw_X = (iter(d.items()) for d in [{'foo': 1, 'bar': 2}, {'baz': 3, 'quux': 4, 'foo': -1}])
    feature_hasher = FeatureHasher(n_features=16, input_type='pair')
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    assert [1, 2] == x1_nz
    assert [1, 3, 4] == x2_nz