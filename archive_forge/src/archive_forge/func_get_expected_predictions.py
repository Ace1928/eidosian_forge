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
def get_expected_predictions(X, y, cv, classes, est, method):
    expected_predictions = np.zeros([len(y), classes])
    func = getattr(est, method)
    for train, test in cv.split(X, y):
        est.fit(X[train], y[train])
        expected_predictions_ = func(X[test])
        if method == 'predict_proba':
            exp_pred_test = np.zeros((len(test), classes))
        else:
            exp_pred_test = np.full((len(test), classes), np.finfo(expected_predictions.dtype).min)
        exp_pred_test[:, est.classes_] = expected_predictions_
        expected_predictions[test] = exp_pred_test
    return expected_predictions