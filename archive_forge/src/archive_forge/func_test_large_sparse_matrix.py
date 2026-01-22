import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_large_sparse_matrix(solver, global_random_seed, csr_container):
    X = csr_container(sparse.rand(20, 10, random_state=global_random_seed))
    for attr in ['indices', 'indptr']:
        setattr(X, attr, getattr(X, attr).astype('int64'))
    rng = np.random.RandomState(global_random_seed)
    y = rng.randint(2, size=X.shape[0])
    if solver in ['liblinear', 'sag', 'saga']:
        msg = 'Only sparse matrices with 32-bit integer indices'
        with pytest.raises(ValueError, match=msg):
            LogisticRegression(solver=solver).fit(X, y)
    else:
        LogisticRegression(solver=solver).fit(X, y)