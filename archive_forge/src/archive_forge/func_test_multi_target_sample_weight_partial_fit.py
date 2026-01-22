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
def test_multi_target_sample_weight_partial_fit():
    X = [[1, 2, 3], [4, 5, 6]]
    y = [[3.141, 2.718], [2.718, 3.141]]
    w = [2.0, 1.0]
    rgr_w = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    rgr_w.partial_fit(X, y, w)
    w = [2.0, 2.0]
    rgr = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    rgr.partial_fit(X, y, w)
    assert rgr.predict(X)[0][0] != rgr_w.predict(X)[0][0]