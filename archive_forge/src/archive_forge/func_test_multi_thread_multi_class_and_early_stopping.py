import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
def test_multi_thread_multi_class_and_early_stopping():
    clf = SGDClassifier(alpha=0.001, tol=0.001, max_iter=1000, early_stopping=True, n_iter_no_change=100, random_state=0, n_jobs=2)
    clf.fit(iris.data, iris.target)
    assert clf.n_iter_ > clf.n_iter_no_change
    assert clf.n_iter_ < clf.n_iter_no_change + 20
    assert clf.score(iris.data, iris.target) > 0.8