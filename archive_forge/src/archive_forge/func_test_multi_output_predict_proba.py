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
def test_multi_output_predict_proba():
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
    param = {'loss': ('hinge', 'log_loss', 'modified_huber')}

    def custom_scorer(estimator, X, y):
        if hasattr(estimator, 'predict_proba'):
            return 1.0
        else:
            return 0.0
    grid_clf = GridSearchCV(sgd_linear_clf, param_grid=param, scoring=custom_scorer, cv=3, error_score='raise')
    multi_target_linear = MultiOutputClassifier(grid_clf)
    multi_target_linear.fit(X, y)
    multi_target_linear.predict_proba(X)
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    multi_target_linear.fit(X, y)
    inner2_msg = "probability estimates are not available for loss='hinge'"
    inner1_msg = "'SGDClassifier' has no attribute 'predict_proba'"
    outer_msg = "'MultiOutputClassifier' has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        multi_target_linear.predict_proba(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner1_msg in str(exec_info.value.__cause__)
    assert isinstance(exec_info.value.__cause__.__cause__, AttributeError)
    assert inner2_msg in str(exec_info.value.__cause__.__cause__)