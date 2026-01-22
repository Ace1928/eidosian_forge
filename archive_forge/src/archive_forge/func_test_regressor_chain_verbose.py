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
def test_regressor_chain_verbose(capsys):
    X, y = make_regression(n_samples=125, n_targets=3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pattern = '\\[Chain\\].*\\(1 of 3\\) Processing order 1, total=.*\\n\\[Chain\\].*\\(2 of 3\\) Processing order 0, total=.*\\n\\[Chain\\].*\\(3 of 3\\) Processing order 2, total=.*\\n$'
    regressor = RegressorChain(LinearRegression(), order=[1, 0, 2], random_state=0, verbose=True)
    regressor.fit(X_train, y_train)
    assert re.match(pattern, capsys.readouterr()[0])