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
def test_dict_learning_reconstruction_parallel():
    n_components = 12
    dico = DictionaryLearning(n_components, transform_algorithm='omp', transform_alpha=0.001, random_state=0, n_jobs=4)
    code = dico.fit(X).transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X)
    dico.set_params(transform_algorithm='lasso_lars')
    code = dico.transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)