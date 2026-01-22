from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_dict_vectorizer_get_feature_names_out():
    """Check that integer feature names are converted to strings in
    feature_names_out."""
    X = [{1: 2, 3: 4}, {2: 4}]
    dv = DictVectorizer(sparse=False).fit(X)
    feature_names = dv.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object
    assert_array_equal(feature_names, ['1', '2', '3'])