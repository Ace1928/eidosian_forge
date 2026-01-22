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
@pytest.mark.parametrize('y_true, y_score', [([0, 0, 1, 2], np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])), ([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0.1, 0.1, 0.4, 0.5, 0.6, 0.6, 0.9, 0.9, 1, 1])])
def test_average_precision_score_duplicate_values(y_true, y_score):
    """
    Duplicate values with precision-recall require a different
    processing than when computing the AUC of a ROC, because the
    precision-recall curve is a decreasing curve
    The following situation corresponds to a perfect
    test statistic, the average_precision_score should be 1.
    """
    assert average_precision_score(y_true, y_score) == 1