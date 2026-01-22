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
@ignore_warnings
def test_precision_recall_f1_score_multilabel_2():
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)
    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.25)
    assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='micro'), (1 + 4) * p * r / (4 * p + r))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='macro')
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.125)
    assert_almost_equal(f, 2 / 12)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='macro'), np.mean(f2))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    assert_almost_equal(p, 2 / 4)
    assert_almost_equal(r, 1 / 4)
    assert_almost_equal(f, 2 / 3 * 2 / 4)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='weighted'), np.average(f2, weights=support))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='samples')
    assert_almost_equal(p, 1 / 6)
    assert_almost_equal(r, 1 / 6)
    assert_almost_equal(f, 2 / 4 * 1 / 3)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='samples'), 0.1666, 2)