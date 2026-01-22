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
def test_dict_learning_nonzero_coefs():
    n_components = 4
    dico = DictionaryLearning(n_components, transform_algorithm='lars', transform_n_nonzero_coefs=3, random_state=0)
    code = dico.fit(X).transform(X[np.newaxis, 1])
    assert len(np.flatnonzero(code)) == 3
    dico.set_params(transform_algorithm='omp')
    code = dico.transform(X[np.newaxis, 1])
    assert len(np.flatnonzero(code)) == 3