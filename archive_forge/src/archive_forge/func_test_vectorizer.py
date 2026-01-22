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
def test_vectorizer():
    train_data = iter(ALL_FOOD_DOCS[:-1])
    test_data = [ALL_FOOD_DOCS[-1]]
    n_train = len(ALL_FOOD_DOCS) - 1
    v1 = CountVectorizer(max_df=0.5)
    counts_train = v1.fit_transform(train_data)
    if hasattr(counts_train, 'tocsr'):
        counts_train = counts_train.tocsr()
    assert counts_train[0, v1.vocabulary_['pizza']] == 2
    v2 = CountVectorizer(vocabulary=v1.vocabulary_)
    for v in (v1, v2):
        counts_test = v.transform(test_data)
        if hasattr(counts_test, 'tocsr'):
            counts_test = counts_test.tocsr()
        vocabulary = v.vocabulary_
        assert counts_test[0, vocabulary['salad']] == 1
        assert counts_test[0, vocabulary['tomato']] == 1
        assert counts_test[0, vocabulary['water']] == 1
        assert 'the' not in vocabulary
        assert 'copyright' not in vocabulary
        assert counts_test[0, vocabulary['coke']] == 0
        assert counts_test[0, vocabulary['burger']] == 0
        assert counts_test[0, vocabulary['beer']] == 0
        assert counts_test[0, vocabulary['pizza']] == 0
    t1 = TfidfTransformer(norm='l1')
    tfidf = t1.fit(counts_train).transform(counts_train).toarray()
    assert len(t1.idf_) == len(v1.vocabulary_)
    assert tfidf.shape == (n_train, len(v1.vocabulary_))
    tfidf_test = t1.transform(counts_test).toarray()
    assert tfidf_test.shape == (len(test_data), len(v1.vocabulary_))
    t2 = TfidfTransformer(norm='l1', use_idf=False)
    tf = t2.fit(counts_train).transform(counts_train).toarray()
    assert not hasattr(t2, 'idf_')
    t3 = TfidfTransformer(use_idf=True)
    with pytest.raises(ValueError):
        t3.transform(counts_train)
    assert_array_almost_equal(np.sum(tf, axis=1), [1.0] * n_train)
    train_data = iter(ALL_FOOD_DOCS[:-1])
    tv = TfidfVectorizer(norm='l1')
    tv.max_df = v1.max_df
    tfidf2 = tv.fit_transform(train_data).toarray()
    assert not tv.fixed_vocabulary_
    assert_array_almost_equal(tfidf, tfidf2)
    tfidf_test2 = tv.transform(test_data).toarray()
    assert_array_almost_equal(tfidf_test, tfidf_test2)
    v3 = CountVectorizer(vocabulary=None)
    with pytest.raises(ValueError):
        v3.transform(train_data)
    v3.set_params(strip_accents='ascii', lowercase=False)
    processor = v3.build_preprocessor()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = strip_accents_ascii(text)
    result = processor(text)
    assert expected == result
    v3.set_params(strip_accents='_gabbledegook_', preprocessor=None)
    with pytest.raises(ValueError):
        v3.build_preprocessor()
    v3.set_params = '_invalid_analyzer_type_'
    with pytest.raises(ValueError):
        v3.build_analyzer()