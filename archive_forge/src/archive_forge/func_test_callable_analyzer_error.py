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
@pytest.mark.parametrize('Estimator', [CountVectorizer, TfidfVectorizer, HashingVectorizer])
@pytest.mark.parametrize('input_type, err_type, err_msg', [('filename', FileNotFoundError, ''), ('file', AttributeError, "'str' object has no attribute 'read'")])
def test_callable_analyzer_error(Estimator, input_type, err_type, err_msg):
    if issubclass(Estimator, HashingVectorizer) and IS_PYPY:
        pytest.xfail('HashingVectorizer is not supported on PyPy')
    data = ['this is text, not file or filename']
    with pytest.raises(err_type, match=err_msg):
        Estimator(analyzer=lambda x: x.split(), input=input_type).fit_transform(data)