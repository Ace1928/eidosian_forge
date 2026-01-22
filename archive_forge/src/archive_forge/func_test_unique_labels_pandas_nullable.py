from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
def test_unique_labels_pandas_nullable(dtype):
    """Checks that unique_labels work with pandas nullable dtypes.

    Non-regression test for gh-25634.
    """
    pd = pytest.importorskip('pandas')
    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype='int64')
    labels = unique_labels(y_true, y_predicted)
    assert_array_equal(labels, [0, 1])