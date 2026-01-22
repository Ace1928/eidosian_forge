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
def test_prf_warnings():
    f, w = (precision_recall_fscore_support, UndefinedMetricWarning)
    for average in [None, 'weighted', 'macro']:
        msg = 'Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.'
        with pytest.warns(w, match=msg):
            f([0, 1, 2], [1, 1, 2], average=average)
        msg = 'Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.'
        with pytest.warns(w, match=msg):
            f([1, 1, 2], [0, 1, 2], average=average)
    msg = 'Precision is ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [1, 0]]), np.array([[1, 0], [0, 0]]), average='samples')
    msg = 'Recall is ill-defined and being set to 0.0 in samples with no true labels. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [1, 0]]), average='samples')
    msg = 'Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), average='micro')
    msg = 'Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f(np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), average='micro')
    msg = 'Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f([1, 1], [-1, -1], average='binary')
    msg = 'Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.'
    with pytest.warns(w, match=msg):
        f([-1, -1], [1, 1], average='binary')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        precision_recall_fscore_support([0, 0], [0, 0], average='binary')
        msg = 'F-score is ill-defined and being set to 0.0 due to no true nor predicted samples. Use `zero_division` parameter to control this behavior.'
        assert str(record.pop().message) == msg
        msg = 'Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.'
        assert str(record.pop().message) == msg
        msg = 'Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.'
        assert str(record.pop().message) == msg