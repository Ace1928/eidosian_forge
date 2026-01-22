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
def test_precision_recall_f1_score_multiclass():
    y_true, y_pred, _ = make_prediction(binary=False)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
    assert_array_almost_equal(r, [0.79, 0.09, 0.9], 2)
    assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
    assert_array_equal(s, [24, 31, 20])
    ps = precision_score(y_true, y_pred, pos_label=1, average='micro')
    assert_array_almost_equal(ps, 0.53, 2)
    rs = recall_score(y_true, y_pred, average='micro')
    assert_array_almost_equal(rs, 0.53, 2)
    fs = f1_score(y_true, y_pred, average='micro')
    assert_array_almost_equal(fs, 0.53, 2)
    ps = precision_score(y_true, y_pred, average='macro')
    assert_array_almost_equal(ps, 0.53, 2)
    rs = recall_score(y_true, y_pred, average='macro')
    assert_array_almost_equal(rs, 0.6, 2)
    fs = f1_score(y_true, y_pred, average='macro')
    assert_array_almost_equal(fs, 0.51, 2)
    ps = precision_score(y_true, y_pred, average='weighted')
    assert_array_almost_equal(ps, 0.51, 2)
    rs = recall_score(y_true, y_pred, average='weighted')
    assert_array_almost_equal(rs, 0.53, 2)
    fs = f1_score(y_true, y_pred, average='weighted')
    assert_array_almost_equal(fs, 0.47, 2)
    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average='samples')
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred, average='samples')
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred, average='samples')
    with pytest.raises(ValueError):
        fbeta_score(y_true, y_pred, average='samples', beta=0.5)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 2, 1], average=None)
    assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
    assert_array_almost_equal(r, [0.79, 0.9, 0.1], 2)
    assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
    assert_array_equal(s, [24, 20, 31])