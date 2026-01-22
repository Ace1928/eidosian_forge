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
def test_countvectorizer_custom_token_pattern():
    """Check `get_feature_names_out()` when a custom token pattern is passed.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12971
    """
    corpus = ['This is the 1st document in my corpus.', 'This document is the 2nd sample.', 'And this is the 3rd one.', 'Is this the 4th document?']
    token_pattern = '[0-9]{1,3}(?:st|nd|rd|th)\\s\\b(\\w{2,})\\b'
    vectorizer = CountVectorizer(token_pattern=token_pattern)
    vectorizer.fit_transform(corpus)
    expected = ['document', 'one', 'sample']
    feature_names_out = vectorizer.get_feature_names_out()
    assert_array_equal(feature_names_out, expected)