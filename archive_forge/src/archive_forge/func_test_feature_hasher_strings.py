import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_feature_hasher_strings():
    raw_X = [['foo', 'bar', 'baz', 'foo'.encode('ascii')], ['bar'.encode('ascii'), 'baz', 'quux']]
    for lg_n_features in (7, 9, 11, 16, 22):
        n_features = 2 ** lg_n_features
        it = (x for x in raw_X)
        feature_hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False)
        X = feature_hasher.transform(it)
        assert X.shape[0] == len(raw_X)
        assert X.shape[1] == n_features
        assert X[0].sum() == 4
        assert X[1].sum() == 3
        assert X.nnz == 6