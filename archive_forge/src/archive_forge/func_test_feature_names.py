import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_feature_names():
    cv = CountVectorizer(max_df=0.5)
    with pytest.raises(ValueError):
        cv.get_feature_names_out()
    assert not cv.fixed_vocabulary_
    X = cv.fit_transform(ALL_FOOD_DOCS)
    n_samples, n_features = X.shape
    assert len(cv.vocabulary_) == n_features
    feature_names = cv.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object
    assert len(feature_names) == n_features
    assert_array_equal(['beer', 'burger', 'celeri', 'coke', 'pizza', 'salad', 'sparkling', 'tomato', 'water'], feature_names)
    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)
    vocab = ['beer', 'burger', 'celeri', 'coke', 'pizza', 'salad', 'sparkling', 'tomato', 'water']
    cv = CountVectorizer(vocabulary=vocab)
    feature_names = cv.get_feature_names_out()
    assert_array_equal(['beer', 'burger', 'celeri', 'coke', 'pizza', 'salad', 'sparkling', 'tomato', 'water'], feature_names)
    assert cv.fixed_vocabulary_
    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)