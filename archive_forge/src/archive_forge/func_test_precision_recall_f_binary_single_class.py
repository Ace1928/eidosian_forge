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
@ignore_warnings
def test_precision_recall_f_binary_single_class():
    assert 1.0 == precision_score([1, 1], [1, 1])
    assert 1.0 == recall_score([1, 1], [1, 1])
    assert 1.0 == f1_score([1, 1], [1, 1])
    assert 1.0 == fbeta_score([1, 1], [1, 1], beta=0)
    assert 0.0 == precision_score([-1, -1], [-1, -1])
    assert 0.0 == recall_score([-1, -1], [-1, -1])
    assert 0.0 == f1_score([-1, -1], [-1, -1])
    assert 0.0 == fbeta_score([-1, -1], [-1, -1], beta=float('inf'))
    assert fbeta_score([-1, -1], [-1, -1], beta=float('inf')) == pytest.approx(fbeta_score([-1, -1], [-1, -1], beta=100000.0))