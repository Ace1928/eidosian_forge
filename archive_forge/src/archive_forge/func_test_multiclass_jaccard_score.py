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
def test_multiclass_jaccard_score(recwarn):
    y_true = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat', 'bird', 'bird']
    y_pred = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird', 'bird', 'cat']
    labels = ['ant', 'bird', 'cat']
    lb = LabelBinarizer()
    lb.fit(labels)
    y_true_bin = lb.transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
    bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
    multi_labels_list = [['ant', 'bird'], ['ant', 'cat'], ['cat', 'bird'], ['ant'], ['bird'], ['cat'], None]
    bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]
    for average in ('macro', 'weighted', 'micro', None):
        for m_label, b_label in zip(multi_labels_list, bin_labels_list):
            assert_almost_equal(multi_jaccard_score(average=average, labels=m_label), bin_jaccard_score(average=average, labels=b_label))
    y_true = np.array([[0, 0], [0, 0], [0, 0]])
    y_pred = np.array([[0, 0], [0, 0], [0, 0]])
    with ignore_warnings():
        assert jaccard_score(y_true, y_pred, average='weighted') == 0
    assert not list(recwarn)