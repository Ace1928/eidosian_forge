from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
@pytest.mark.parametrize('sparse', (True, False))
@pytest.mark.parametrize('dtype', (int, np.float32, np.int16))
@pytest.mark.parametrize('sort', (True, False))
@pytest.mark.parametrize('iterable', (True, False))
def test_dictvectorizer(sparse, dtype, sort, iterable):
    D = [{'foo': 1, 'bar': 3}, {'bar': 4, 'baz': 2}, {'bar': 1, 'quux': 1, 'quuux': 2}]
    v = DictVectorizer(sparse=sparse, dtype=dtype, sort=sort)
    X = v.fit_transform(iter(D) if iterable else D)
    assert sp.issparse(X) == sparse
    assert X.shape == (3, 5)
    assert X.sum() == 14
    assert v.inverse_transform(X) == D
    if sparse:
        assert_array_equal(X.toarray(), v.transform(iter(D) if iterable else D).toarray())
    else:
        assert_array_equal(X, v.transform(iter(D) if iterable else D))
    if sort:
        assert v.feature_names_ == sorted(v.feature_names_)