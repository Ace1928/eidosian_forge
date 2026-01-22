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
def test_cross_val_score_predict_groups():
    X, y = make_classification(n_samples=20, n_classes=2, random_state=0)
    clf = SVC(kernel='linear')
    group_cvs = [LeaveOneGroupOut(), LeavePGroupsOut(2), GroupKFold(), GroupShuffleSplit()]
    error_message = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        with pytest.raises(ValueError, match=error_message):
            cross_val_score(estimator=clf, X=X, y=y, cv=cv)
        with pytest.raises(ValueError, match=error_message):
            cross_val_predict(estimator=clf, X=X, y=y, cv=cv)