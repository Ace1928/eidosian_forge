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
def test_hinge_loss_multiclass_no_consistent_pred_decision_shape():
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    error_message = 'The shape of pred_decision cannot be 1d arraywith a multiclass target. pred_decision shape must be (n_samples, n_classes), that is (7, 3). Got: (7,)'
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision)
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    error_message = 'The shape of pred_decision is not consistent with the number of classes. With a multiclass target, pred_decision shape must be (n_samples, n_classes), that is (7, 3). Got: (7, 2)'
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision, labels=labels)