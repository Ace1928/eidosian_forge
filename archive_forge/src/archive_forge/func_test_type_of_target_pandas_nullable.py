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
def test_type_of_target_pandas_nullable():
    """Check that type_of_target works with pandas nullable dtypes."""
    pd = pytest.importorskip('pandas')
    for dtype in ['Int32', 'Float32']:
        y_true = pd.Series([1, 0, 2, 3, 4], dtype=dtype)
        assert type_of_target(y_true) == 'multiclass'
        y_true = pd.Series([1, 0, 1, 0], dtype=dtype)
        assert type_of_target(y_true) == 'binary'
    y_true = pd.DataFrame([[1.4, 3.1], [3.1, 1.4]], dtype='Float32')
    assert type_of_target(y_true) == 'continuous-multioutput'
    y_true = pd.DataFrame([[0, 1], [1, 1]], dtype='Int32')
    assert type_of_target(y_true) == 'multilabel-indicator'
    y_true = pd.DataFrame([[1, 2], [3, 1]], dtype='Int32')
    assert type_of_target(y_true) == 'multiclass-multioutput'