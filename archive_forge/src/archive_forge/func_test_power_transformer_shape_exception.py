import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('method', ['box-cox', 'yeo-johnson'])
def test_power_transformer_shape_exception(method):
    pt = PowerTransformer(method=method)
    X = np.abs(X_2d)
    pt.fit(X)
    wrong_shape_message = 'X has \\d+ features, but PowerTransformer is expecting \\d+ features'
    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.transform(X[:, 0:1])
    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.inverse_transform(X[:, 0:1])