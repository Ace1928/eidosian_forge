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
@pytest.mark.parametrize('transform_algorithm', ('lasso_lars', 'lasso_cd', 'lars', 'threshold', 'omp'))
@pytest.mark.parametrize('data_type', (np.float32, np.float64))
def test_sparse_coder_dtype_match(data_type, transform_algorithm):
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)
    coder = SparseCoder(dictionary.astype(data_type), transform_algorithm=transform_algorithm)
    code = coder.transform(X.astype(data_type))
    assert code.dtype == data_type