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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_multilabel_confusion_matrix_multilabel(csc_container, csr_container):
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    y_true_csr = csr_container(y_true)
    y_pred_csr = csr_container(y_pred)
    y_true_csc = csc_container(y_true)
    y_pred_csc = csc_container(y_pred)
    sample_weight = np.array([2, 1, 3])
    real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
    trues = [y_true, y_true_csr, y_true_csc]
    preds = [y_pred, y_pred_csr, y_pred_csc]
    for y_true_tmp in trues:
        for y_pred_tmp in preds:
            cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
            assert_array_equal(cm, real_cm)
    cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0])
    assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0], samplewise=True)
    assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])
    cm = multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, samplewise=True)
    assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])