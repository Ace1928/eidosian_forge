import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
@pytest.mark.parametrize('method', ('lars', 'cd'))
@pytest.mark.parametrize('data_type, expected_type', ((np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)))
def test_dict_learning_dtype_match(data_type, expected_type, method):
    rng = np.random.RandomState(0)
    n_components = 8
    code, dictionary, _ = dict_learning(X.astype(data_type), n_components=n_components, alpha=1, random_state=rng, method=method)
    assert code.dtype == expected_type
    assert dictionary.dtype == expected_type