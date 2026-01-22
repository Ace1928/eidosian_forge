from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('y, params, type_err, msg_err', [(y_iris, {'estimators': []}, ValueError, "Invalid 'estimators' attribute,"), (y_iris, {'estimators': [('lr', LogisticRegression()), ('svm', SVC(max_iter=50000))], 'stack_method': 'predict_proba'}, ValueError, 'does not implement the method predict_proba'), (y_iris, {'estimators': [('lr', LogisticRegression()), ('cor', NoWeightClassifier())]}, TypeError, 'does not support sample weight'), (y_iris, {'estimators': [('lr', LogisticRegression()), ('cor', LinearSVC(dual='auto', max_iter=50000))], 'final_estimator': NoWeightClassifier()}, TypeError, 'does not support sample weight')])
def test_stacking_classifier_error(y, params, type_err, msg_err):
    with pytest.raises(type_err, match=msg_err):
        clf = StackingClassifier(**params, cv=3)
        clf.fit(scale(X_iris), y, sample_weight=np.ones(X_iris.shape[0]))