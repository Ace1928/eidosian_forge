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
def test_average_binary_jaccard_score(recwarn):
    assert jaccard_score([1], [0], average='binary') == 0.0
    msg = 'Jaccard is ill-defined and being set to 0.0 due to no true or predicted samples'
    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert jaccard_score([0, 0], [0, 0], average='binary') == 0.0
    assert jaccard_score([0], [0], pos_label=0, average='binary') == 1.0
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 1])
    assert_almost_equal(jaccard_score(y_true, y_pred, average='binary'), 3.0 / 4)
    assert_almost_equal(jaccard_score(y_true, y_pred, average='binary', pos_label=0), 1.0 / 2)
    assert not list(recwarn)