import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
def test_dataframe_support(pyplot):
    """Check that passing a dataframe at fit and to the Display does not
    raise warnings.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23311
    """
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(X, columns=['col_x', 'col_y'])
    estimator = LogisticRegression().fit(df, y)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        DecisionBoundaryDisplay.from_estimator(estimator, df, response_method='predict')