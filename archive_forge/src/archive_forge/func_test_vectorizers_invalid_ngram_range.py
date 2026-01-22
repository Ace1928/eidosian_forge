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
@pytest.mark.parametrize('vec', [HashingVectorizer(ngram_range=(2, 1)), CountVectorizer(ngram_range=(2, 1)), TfidfVectorizer(ngram_range=(2, 1))])
def test_vectorizers_invalid_ngram_range(vec):
    invalid_range = vec.ngram_range
    message = re.escape(f'Invalid value for ngram_range={invalid_range} lower boundary larger than the upper boundary.')
    if isinstance(vec, HashingVectorizer) and IS_PYPY:
        pytest.xfail(reason='HashingVectorizer is not supported on PyPy')
    with pytest.raises(ValueError, match=message):
        vec.fit(['good news everyone'])
    with pytest.raises(ValueError, match=message):
        vec.fit_transform(['good news everyone'])
    if isinstance(vec, HashingVectorizer):
        with pytest.raises(ValueError, match=message):
            vec.transform(['good news everyone'])