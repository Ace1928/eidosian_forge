import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.datasets import (
from sklearn.ensemble import (
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
@pytest.mark.parametrize('Ensemble, Estimator, X, y', [(StackingClassifier, LogisticRegression, X, y), (StackingRegressor, LinearRegression, X_r, y_r), (VotingClassifier, LogisticRegression, X, y), (VotingRegressor, LinearRegression, X_r, y_r)])
def test_heterogeneous_ensemble_support_missing_values(Ensemble, Estimator, X, y):
    X = X.copy()
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    pipe = make_pipeline(SimpleImputer(), Estimator())
    ensemble = Ensemble(estimators=[('pipe1', pipe), ('pipe2', pipe)])
    ensemble.fit(X, y).score(X, y)