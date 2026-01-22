import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('y, params, err_msg', [([2, 1, 2, 2], {'random_state': 0}, 'Constant.*has to be specified'), ([2, 1, 2, 2], {'constant': [2, 0]}, 'Constant.*should have shape'), (np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]), {'constant': 2}, 'Constant.*should have shape'), ([2, 1, 2, 2], {'constant': 'my-constant'}, 'constant=my-constant.*Possible values.*\\[1, 2]'), (np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]), {'constant': [2, 'unknown']}, "constant=\\[2, 'unknown'].*Possible values.*\\[1, 2]")], ids=['no-constant', 'too-many-constant', 'not-enough-output', 'single-output', 'multi-output'])
def test_constant_strategy_exceptions(y, params, err_msg):
    X = [[0], [0], [0], [0]]
    clf = DummyClassifier(strategy='constant', **params)
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)