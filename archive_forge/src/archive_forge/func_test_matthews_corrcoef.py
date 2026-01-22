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
def test_matthews_corrcoef():
    rng = np.random.RandomState(0)
    y_true = ['a' if i == 0 else 'b' for i in rng.randint(0, 2, size=20)]
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    y_true_inv = ['b' if i == 'a' else 'a' for i in y_true]
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv), -1)
    y_true_inv2 = label_binarize(y_true, classes=['a', 'b'])
    y_true_inv2 = np.where(y_true_inv2, 'a', 'b')
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv2), -1)
    assert_almost_equal(matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)
    assert_almost_equal(matthews_corrcoef(y_true, ['a'] * len(y_true)), 0.0)
    y_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    y_2 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)
    mask = [1] * 10 + [0] * 10
    with pytest.raises(AssertionError):
        assert_almost_equal(matthews_corrcoef(y_1, y_2, sample_weight=mask), 0.0)