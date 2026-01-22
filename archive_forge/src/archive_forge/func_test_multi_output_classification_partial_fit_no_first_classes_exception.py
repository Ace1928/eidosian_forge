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
def test_multi_output_classification_partial_fit_no_first_classes_exception():
    sgd_linear_clf = SGDClassifier(loss='log_loss', random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    msg = 'classes must be passed on the first call to partial_fit.'
    with pytest.raises(ValueError, match=msg):
        multi_target_linear.partial_fit(X, y)