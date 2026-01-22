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
@pytest.mark.parametrize('Vectorizer', [CountVectorizer, HashingVectorizer, TfidfVectorizer])
@pytest.mark.parametrize('stop_words, tokenizer, preprocessor, ngram_range, token_pattern,analyzer, unused_name, ovrd_name, ovrd_msg', [(["you've", "you'll"], None, None, (1, 1), None, 'char', "'stop_words'", "'analyzer'", "!= 'word'"), (None, lambda s: s.split(), None, (1, 1), None, 'char', "'tokenizer'", "'analyzer'", "!= 'word'"), (None, lambda s: s.split(), None, (1, 1), '\\w+', 'word', "'token_pattern'", "'tokenizer'", 'is not None'), (None, None, lambda s: s.upper(), (1, 1), '\\w+', lambda s: s.upper(), "'preprocessor'", "'analyzer'", 'is callable'), (None, None, None, (1, 2), None, lambda s: s.upper(), "'ngram_range'", "'analyzer'", 'is callable'), (None, None, None, (1, 1), '\\w+', 'char', "'token_pattern'", "'analyzer'", "!= 'word'")])
def test_unused_parameters_warn(Vectorizer, stop_words, tokenizer, preprocessor, ngram_range, token_pattern, analyzer, unused_name, ovrd_name, ovrd_msg):
    train_data = JUNK_FOOD_DOCS
    vect = Vectorizer()
    vect.set_params(stop_words=stop_words, tokenizer=tokenizer, preprocessor=preprocessor, ngram_range=ngram_range, token_pattern=token_pattern, analyzer=analyzer)
    msg = 'The parameter %s will not be used since %s %s' % (unused_name, ovrd_name, ovrd_msg)
    with pytest.warns(UserWarning, match=msg):
        vect.fit(train_data)