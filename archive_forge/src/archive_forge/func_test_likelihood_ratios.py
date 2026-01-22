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
def test_likelihood_ratios():
    y_true = np.array([1] * 3 + [0] * 17)
    y_pred = np.array([1] * 2 + [0] * 10 + [1] * 8)
    pos, neg = class_likelihood_ratios(y_true, y_pred)
    assert_allclose(pos, 34 / 24)
    assert_allclose(neg, 17 / 27)
    pos, neg = class_likelihood_ratios(y_true, y_true)
    assert_array_equal(pos, np.nan * 2)
    assert_allclose(neg, np.zeros(2), rtol=1e-12)
    sample_weight = np.array([1.0] * 15 + [0.0] * 5)
    pos, neg = class_likelihood_ratios(y_true, y_pred, sample_weight=sample_weight)
    assert_allclose(pos, 24 / 9)
    assert_allclose(neg, 12 / 27)