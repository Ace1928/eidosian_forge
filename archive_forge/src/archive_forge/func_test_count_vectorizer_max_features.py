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
def test_count_vectorizer_max_features():
    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)
    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    features_1 = cv_1.get_feature_names_out()
    features_3 = cv_3.get_feature_names_out()
    features_None = cv_None.get_feature_names_out()
    assert 7 == counts_1.max()
    assert 7 == counts_3.max()
    assert 7 == counts_None.max()
    assert 'the' == features_1[np.argmax(counts_1)]
    assert 'the' == features_3[np.argmax(counts_3)]
    assert 'the' == features_None[np.argmax(counts_None)]