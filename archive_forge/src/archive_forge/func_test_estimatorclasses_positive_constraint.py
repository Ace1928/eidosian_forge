import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_estimatorclasses_positive_constraint():
    default_parameter = {'fit_intercept': False}
    estimator_parameter_map = {'LassoLars': {'alpha': 0.1}, 'LassoLarsCV': {}, 'LassoLarsIC': {}}
    for estname in estimator_parameter_map:
        params = default_parameter.copy()
        params.update(estimator_parameter_map[estname])
        estimator = getattr(linear_model, estname)(positive=False, **params)
        estimator.fit(X, y)
        assert estimator.coef_.min() < 0
        estimator = getattr(linear_model, estname)(positive=True, **params)
        estimator.fit(X, y)
        assert min(estimator.coef_) >= 0