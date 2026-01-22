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
def test_confusion_matrix_binary():
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        assert_array_equal(cm, [[22, 3], [8, 17]])
        tp, fp, fn, tn = cm.flatten()
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        true_mcc = 0 if den == 0 else num / den
        mcc = matthews_corrcoef(y_true, y_pred)
        assert_array_almost_equal(mcc, true_mcc, decimal=2)
        assert_array_almost_equal(mcc, 0.57, decimal=2)
    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])