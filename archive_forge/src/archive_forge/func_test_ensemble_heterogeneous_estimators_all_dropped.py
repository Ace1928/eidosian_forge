import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.datasets import (
from sklearn.ensemble import (
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
@pytest.mark.parametrize('X, y, estimator', [(*make_classification(n_samples=10), StackingClassifier(estimators=[('lr', LogisticRegression())])), (*make_classification(n_samples=10), VotingClassifier(estimators=[('lr', LogisticRegression())])), (*make_regression(n_samples=10), StackingRegressor(estimators=[('lr', LinearRegression())])), (*make_regression(n_samples=10), VotingRegressor(estimators=[('lr', LinearRegression())]))], ids=['stacking-classifier', 'voting-classifier', 'stacking-regressor', 'voting-regressor'])
def test_ensemble_heterogeneous_estimators_all_dropped(X, y, estimator):
    estimator.set_params(lr='drop')
    with pytest.raises(ValueError, match='All estimators are dropped.'):
        estimator.fit(X, y)