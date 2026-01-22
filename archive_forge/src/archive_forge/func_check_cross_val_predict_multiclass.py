import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def check_cross_val_predict_multiclass(est, X, y, method):
    """Helper for tests of cross_val_predict with multiclass classification"""
    cv = KFold(n_splits=3, shuffle=False)
    float_min = np.finfo(np.float64).min
    default_values = {'decision_function': float_min, 'predict_log_proba': float_min, 'predict_proba': 0}
    expected_predictions = np.full((len(X), len(set(y))), default_values[method], dtype=np.float64)
    _, y_enc = np.unique(y, return_inverse=True)
    for train, test in cv.split(X, y_enc):
        est = clone(est).fit(X[train], y_enc[train])
        fold_preds = getattr(est, method)(X[test])
        i_cols_fit = np.unique(y_enc[train])
        expected_predictions[np.ix_(test, i_cols_fit)] = fold_preds
    for tg in [y, y + 1, y - 2, y.astype('str')]:
        assert_allclose(cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions)