import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('response_method', ['predict_proba', 'predict_log_proba', 'transform'])
def test_classifier_undefined_methods(response_method):
    clf = PassiveAggressiveClassifier(max_iter=100)
    with pytest.raises(AttributeError):
        getattr(clf, response_method)