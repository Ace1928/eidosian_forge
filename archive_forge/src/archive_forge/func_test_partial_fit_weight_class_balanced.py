import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_partial_fit_weight_class_balanced():
    clf = PassiveAggressiveClassifier(class_weight='balanced', max_iter=100)
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=np.unique(y))