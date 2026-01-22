import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.datasets import (
from sklearn.ensemble import (
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
@pytest.mark.parametrize('X, y, Ensemble', [(*make_classification(n_samples=10), StackingClassifier), (*make_classification(n_samples=10), VotingClassifier), (*make_regression(n_samples=10), StackingRegressor), (*make_regression(n_samples=10), VotingRegressor)])
def test_ensemble_heterogeneous_estimators_name_validation(X, y, Ensemble):
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [('lr__', LogisticRegression())]
    else:
        estimators = [('lr__', LinearRegression())]
    ensemble = Ensemble(estimators=estimators)
    err_msg = "Estimator names must not contain __: got \\['lr__'\\]"
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [('lr', LogisticRegression()), ('lr', LogisticRegression())]
    else:
        estimators = [('lr', LinearRegression()), ('lr', LinearRegression())]
    ensemble = Ensemble(estimators=estimators)
    err_msg = "Names provided are not unique: \\['lr', 'lr'\\]"
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [('estimators', LogisticRegression())]
    else:
        estimators = [('estimators', LinearRegression())]
    ensemble = Ensemble(estimators=estimators)
    err_msg = 'Estimator names conflict with constructor arguments'
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)