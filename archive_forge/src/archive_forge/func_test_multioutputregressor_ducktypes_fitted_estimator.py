import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_multioutputregressor_ducktypes_fitted_estimator():
    """Test that MultiOutputRegressor checks the fitted estimator for
    predict. Non-regression test for #16549."""
    X, y = load_linnerud(return_X_y=True)
    stacker = StackingRegressor(estimators=[('sgd', SGDRegressor(random_state=1))], final_estimator=Ridge(), cv=2)
    reg = MultiOutputRegressor(estimator=stacker).fit(X, y)
    reg.predict(X)