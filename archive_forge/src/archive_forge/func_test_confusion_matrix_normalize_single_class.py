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
def test_confusion_matrix_normalize_single_class():
    y_test = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
    cm_true = confusion_matrix(y_test, y_pred, normalize='true')
    assert cm_true.sum() == pytest.approx(2.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        cm_pred = confusion_matrix(y_test, y_pred, normalize='pred')
    assert cm_pred.sum() == pytest.approx(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        confusion_matrix(y_pred, y_test, normalize='true')