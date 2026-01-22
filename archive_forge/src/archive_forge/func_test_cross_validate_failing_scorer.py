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
@pytest.mark.filterwarnings('ignore:lbfgs failed to converge')
@pytest.mark.parametrize('error_score', [np.nan, 0, 'raise'])
@pytest.mark.parametrize('return_train_score', [True, False])
@pytest.mark.parametrize('with_multimetric', [False, True])
def test_cross_validate_failing_scorer(error_score, return_train_score, with_multimetric):
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=5).fit(X, y)
    error_msg = 'This scorer is supposed to fail!!!'
    failing_scorer = partial(_failing_scorer, error_msg=error_msg)
    if with_multimetric:
        non_failing_scorer = make_scorer(mean_squared_error)
        scoring = {'score_1': failing_scorer, 'score_2': non_failing_scorer, 'score_3': failing_scorer}
    else:
        scoring = failing_scorer
    if error_score == 'raise':
        with pytest.raises(ValueError, match=error_msg):
            cross_validate(clf, X, y, cv=3, scoring=scoring, return_train_score=return_train_score, error_score=error_score)
    else:
        warning_msg = f'Scoring failed. The score on this train-test partition for these parameters will be set to {error_score}'
        with pytest.warns(UserWarning, match=warning_msg):
            results = cross_validate(clf, X, y, cv=3, scoring=scoring, return_train_score=return_train_score, error_score=error_score)
            for key in results:
                if '_score' in key:
                    if '_score_2' in key:
                        for i in results[key]:
                            assert isinstance(i, float)
                    else:
                        assert_allclose(results[key], error_score)