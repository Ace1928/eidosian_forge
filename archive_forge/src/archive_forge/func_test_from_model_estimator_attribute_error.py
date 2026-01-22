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
def test_from_model_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the estimator
    does not implement the `partial_fit` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    from_model = SelectFromModel(estimator=LinearRegression())
    outer_msg = "This 'SelectFromModel' has no attribute 'partial_fit'"
    inner_msg = "'LinearRegression' object has no attribute 'partial_fit'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        from_model.fit(data, y).partial_fit(data)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)