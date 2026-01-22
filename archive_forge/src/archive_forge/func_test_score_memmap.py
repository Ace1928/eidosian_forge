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
def test_score_memmap():
    iris = load_iris()
    X, y = (iris.data, iris.target)
    clf = MockClassifier()
    tf = tempfile.NamedTemporaryFile(mode='wb', delete=False)
    tf.write(b'Hello world!!!!!')
    tf.close()
    scores = np.memmap(tf.name, dtype=np.float64)
    score = np.memmap(tf.name, shape=(), mode='r', dtype=np.float64)
    try:
        cross_val_score(clf, X, y, scoring=lambda est, X, y: score)
        with pytest.raises(ValueError):
            cross_val_score(clf, X, y, scoring=lambda est, X, y: scores)
    finally:
        scores, score = (None, None)
        for _ in range(3):
            try:
                os.unlink(tf.name)
                break
            except OSError:
                sleep(1.0)