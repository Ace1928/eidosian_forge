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
def test_log_loss_pandas_input():
    y_tr = np.array(['ham', 'spam', 'spam', 'ham'])
    y_pr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series
        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        y_true, y_pred = (TrueInputType(y_tr), PredInputType(y_pr))
        with pytest.warns(UserWarning, match='y_pred values do not sum to one'):
            loss = log_loss(y_true, y_pred)
        assert_almost_equal(loss, 1.0383217, decimal=6)