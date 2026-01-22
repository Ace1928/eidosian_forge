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
def test_hashingvectorizer_nan_in_docs():
    message = 'np.nan is an invalid document, expected byte or unicode string.'
    exception = ValueError

    def func():
        hv = HashingVectorizer()
        hv.fit_transform(['hello world', np.nan, 'hello hello'])
    with pytest.raises(exception, match=message):
        func()