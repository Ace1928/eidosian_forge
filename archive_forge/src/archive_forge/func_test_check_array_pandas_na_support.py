import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
@pytest.mark.parametrize('pd_dtype', ['Int8', 'Int16', 'UInt8', 'UInt16', 'Float32', 'Float64'])
@pytest.mark.parametrize('dtype, expected_dtype', [([np.float32, np.float64], np.float32), (np.float64, np.float64), ('numeric', np.float64)])
def test_check_array_pandas_na_support(pd_dtype, dtype, expected_dtype):
    pd = pytest.importorskip('pandas')
    if pd_dtype in {'Float32', 'Float64'}:
        pd = pytest.importorskip('pandas', minversion='1.2')
    X_np = np.array([[1, 2, 3, np.nan, np.nan], [np.nan, np.nan, 8, 4, 6], [1, 2, 3, 4, 5]]).T
    X = pd.DataFrame(X_np, dtype=pd_dtype, columns=['a', 'b', 'c'])
    X['c'] = X['c'].astype('float')
    X_checked = check_array(X, force_all_finite='allow-nan', dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype
    X_checked = check_array(X, force_all_finite=False, dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype
    msg = 'Input contains NaN'
    with pytest.raises(ValueError, match=msg):
        check_array(X, force_all_finite=True)