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
@pytest.mark.parametrize('Estimator', [CountVectorizer, TfidfVectorizer, HashingVectorizer])
def test_stop_word_validation_custom_preprocessor(Estimator):
    data = [{'text': 'some text'}]
    vec = Estimator()
    assert _check_stop_words_consistency(vec) is True
    vec = Estimator(preprocessor=lambda x: x['text'], stop_words=['and'])
    assert _check_stop_words_consistency(vec) == 'error'
    assert _check_stop_words_consistency(vec) is None
    vec.fit_transform(data)

    class CustomEstimator(Estimator):

        def build_preprocessor(self):
            return lambda x: x['text']
    vec = CustomEstimator(stop_words=['and'])
    assert _check_stop_words_consistency(vec) == 'error'
    vec = Estimator(tokenizer=lambda doc: re.compile('\\w{1,}').findall(doc), stop_words=['and'])
    assert _check_stop_words_consistency(vec) is True