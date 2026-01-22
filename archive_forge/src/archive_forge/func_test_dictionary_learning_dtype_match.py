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
@pytest.mark.parametrize('fit_algorithm', ('lars', 'cd'))
@pytest.mark.parametrize('transform_algorithm', ('lasso_lars', 'lasso_cd', 'lars', 'threshold', 'omp'))
@pytest.mark.parametrize('data_type, expected_type', ((np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)))
def test_dictionary_learning_dtype_match(data_type, expected_type, fit_algorithm, transform_algorithm):
    dict_learner = DictionaryLearning(n_components=8, fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm, random_state=0)
    dict_learner.fit(X.astype(data_type))
    assert dict_learner.components_.dtype == expected_type
    assert dict_learner.transform(X.astype(data_type)).dtype == expected_type