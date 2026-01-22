import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('cols, pattern, include, exclude', [(['col_int', 'col_float'], None, np.number, None), (['col_int', 'col_float'], None, None, object), (['col_int', 'col_float'], None, [int, float], None), (['col_str'], None, [object], None), (['col_str'], None, object, None), (['col_float'], None, float, None), (['col_float'], 'at$', [np.number], None), (['col_int'], None, [int], None), (['col_int'], '^col_int', [np.number], None), (['col_float', 'col_str'], 'float|str', None, None), (['col_str'], '^col_s', None, [int]), ([], 'str$', float, None), (['col_int', 'col_float', 'col_str'], None, [np.number, object], None)])
def test_make_column_selector_with_select_dtypes(cols, pattern, include, exclude):
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'col_int': np.array([0, 1, 2], dtype=int), 'col_float': np.array([0.0, 1.0, 2.0], dtype=float), 'col_str': ['one', 'two', 'three']}, columns=['col_int', 'col_float', 'col_str'])
    selector = make_column_selector(dtype_include=include, dtype_exclude=exclude, pattern=pattern)
    assert_array_equal(selector(X_df), cols)