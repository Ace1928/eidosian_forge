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
def test_prf_average_binary_data_non_binary():
    y_true_mc = [1, 2, 3, 3]
    y_pred_mc = [1, 2, 3, 1]
    msg_mc = "Target is multiclass but average='binary'. Please choose another average setting, one of \\[None, 'micro', 'macro', 'weighted'\\]."
    y_true_ind = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
    y_pred_ind = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    msg_ind = "Target is multilabel-indicator but average='binary'. Please choose another average setting, one of \\[None, 'micro', 'macro', 'weighted', 'samples'\\]."
    for y_true, y_pred, msg in [(y_true_mc, y_pred_mc, msg_mc), (y_true_ind, y_pred_ind, msg_ind)]:
        for metric in [precision_score, recall_score, f1_score, partial(fbeta_score, beta=2)]:
            with pytest.raises(ValueError, match=msg):
                metric(y_true, y_pred)