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
@pytest.mark.parametrize('Vectorizer', (CountVectorizer, TfidfVectorizer))
def test_vectorizer_inverse_transform(Vectorizer):
    data = ALL_FOOD_DOCS
    vectorizer = Vectorizer()
    transformed_data = vectorizer.fit_transform(data)
    inversed_data = vectorizer.inverse_transform(transformed_data)
    assert isinstance(inversed_data, list)
    analyze = vectorizer.build_analyzer()
    for doc, inversed_terms in zip(data, inversed_data):
        terms = np.sort(np.unique(analyze(doc)))
        inversed_terms = np.sort(np.unique(inversed_terms))
        assert_array_equal(terms, inversed_terms)
    assert sparse.issparse(transformed_data)
    assert transformed_data.format == 'csr'
    transformed_data2 = transformed_data.toarray()
    inversed_data2 = vectorizer.inverse_transform(transformed_data2)
    for terms, terms2 in zip(inversed_data, inversed_data2):
        assert_array_equal(np.sort(terms), np.sort(terms2))
    transformed_data3 = transformed_data.tocsc()
    inversed_data3 = vectorizer.inverse_transform(transformed_data3)
    for terms, terms3 in zip(inversed_data, inversed_data3):
        assert_array_equal(np.sort(terms), np.sort(terms3))