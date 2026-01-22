import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('scoring', [make_scorer(f1_score, zero_division=np.nan), make_scorer(fbeta_score, beta=2, zero_division=np.nan), make_scorer(precision_score, zero_division=np.nan), make_scorer(recall_score, zero_division=np.nan)])
def test_classification_metric_division_by_zero_nan_validaton(scoring):
    """Check that we validate `np.nan` properly for classification metrics.

    With `n_jobs=2` in cross-validation, the `np.nan` used for the singleton will be
    different in the sub-process and we should not use the `is` operator but
    `math.isnan`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27563
    """
    X, y = datasets.make_classification(random_state=0)
    classifier = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    cross_val_score(classifier, X, y, scoring=scoring, n_jobs=2, error_score='raise')