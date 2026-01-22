import re
import warnings
from unittest.mock import Mock
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
def test_calling_fit_reinitializes():
    est = LinearSVC(dual='auto', random_state=0)
    transformer = SelectFromModel(estimator=est)
    transformer.fit(data, y)
    transformer.set_params(estimator__C=100)
    transformer.fit(data, y)
    assert transformer.estimator_.C == 100