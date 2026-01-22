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
def test_cd_work_on_joblib_memmapped_data(monkeypatch):
    monkeypatch.setattr(sklearn.decomposition._dict_learning, 'Parallel', partial(Parallel, max_nbytes=100))
    rng = np.random.RandomState(0)
    X_train = rng.randn(10, 10)
    dict_learner = DictionaryLearning(n_components=5, random_state=0, n_jobs=2, fit_algorithm='cd', max_iter=50, verbose=True)
    dict_learner.fit(X_train)