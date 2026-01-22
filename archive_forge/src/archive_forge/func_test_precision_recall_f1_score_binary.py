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
def test_precision_recall_f1_score_binary():
    y_true, y_pred, _ = make_prediction(binary=True)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.73, 0.85], 2)
    assert_array_almost_equal(r, [0.88, 0.68], 2)
    assert_array_almost_equal(f, [0.8, 0.76], 2)
    assert_array_equal(s, [25, 25])
    for kwargs, my_assert in [({}, assert_no_warnings), ({'average': 'binary'}, assert_no_warnings)]:
        ps = my_assert(precision_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(ps, 0.85, 2)
        rs = my_assert(recall_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(rs, 0.68, 2)
        fs = my_assert(f1_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(fs, 0.76, 2)
        assert_almost_equal(my_assert(fbeta_score, y_true, y_pred, beta=2, **kwargs), (1 + 2 ** 2) * ps * rs / (2 ** 2 * ps + rs), 2)