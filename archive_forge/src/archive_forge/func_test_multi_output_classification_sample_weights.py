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
def test_multi_output_classification_sample_weights():
    Xw = [[1, 2, 3], [4, 5, 6]]
    yw = [[3, 2], [2, 3]]
    w = np.asarray([2.0, 1.0])
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf_w = MultiOutputClassifier(forest)
    clf_w.fit(Xw, yw, w)
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    y = [[3, 2], [3, 2], [2, 3]]
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf = MultiOutputClassifier(forest)
    clf.fit(X, y)
    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    assert_almost_equal(clf.predict(X_test), clf_w.predict(X_test))