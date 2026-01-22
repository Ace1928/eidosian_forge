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
@fails_if_pypy
def test_vectorizer_unicode():
    document = 'Машинное обучение — обширный подраздел искусственного интеллекта, изучающий методы построения алгоритмов, способных обучаться.'
    vect = CountVectorizer()
    X_counted = vect.fit_transform([document])
    assert X_counted.shape == (1, 12)
    vect = HashingVectorizer(norm=None, alternate_sign=False)
    X_hashed = vect.transform([document])
    assert X_hashed.shape == (1, 2 ** 20)
    assert X_counted.nnz == X_hashed.nnz
    assert_array_equal(np.sort(X_counted.data), np.sort(X_hashed.data))