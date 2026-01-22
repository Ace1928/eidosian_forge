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
@pytest.mark.parametrize('average', ['macro', 'micro', 'weighted', 'samples'])
def test_precision_recall_f1_no_labels_check_warnings(average):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)
    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)
    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None
    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, average=average, beta=1.0)
    assert_almost_equal(fbeta, 0)