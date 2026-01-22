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
def test_unicode_decode_error():
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    text_bytes = text.encode('utf-8')
    wa = CountVectorizer(ngram_range=(1, 2), encoding='ascii').build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        wa(text_bytes)
    ca = CountVectorizer(analyzer='char', ngram_range=(3, 6), encoding='ascii').build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        ca(text_bytes)