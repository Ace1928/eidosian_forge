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
def test_allow_nan_tag_comes_from_estimator():
    allow_nan_est = NaNTag()
    model = SelectFromModel(estimator=allow_nan_est)
    assert model._get_tags()['allow_nan'] is True
    no_nan_est = NoNaNTag()
    model = SelectFromModel(estimator=no_nan_est)
    assert model._get_tags()['allow_nan'] is False