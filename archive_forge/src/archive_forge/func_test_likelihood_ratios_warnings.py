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
@pytest.mark.parametrize('params, warn_msg', [({'y_true': np.array([0, 0, 0, 0, 0, 0]), 'y_pred': np.array([0, 0, 0, 0, 0, 0])}, 'samples of only one class were seen during testing'), ({'y_true': np.array([1, 1, 1, 0, 0, 0]), 'y_pred': np.array([1, 1, 1, 0, 0, 0])}, 'positive_likelihood_ratio ill-defined and being set to nan'), ({'y_true': np.array([1, 1, 1, 0, 0, 0]), 'y_pred': np.array([0, 0, 0, 0, 0, 0])}, 'no samples predicted for the positive class'), ({'y_true': np.array([1, 1, 1, 0, 0, 0]), 'y_pred': np.array([0, 0, 0, 1, 1, 1])}, 'negative_likelihood_ratio ill-defined and being set to nan'), ({'y_true': np.array([0, 0, 0, 0, 0, 0]), 'y_pred': np.array([1, 1, 1, 0, 0, 0])}, 'no samples of the positive class were present in the testing set')])
def test_likelihood_ratios_warnings(params, warn_msg):
    with pytest.warns(UserWarning, match=warn_msg):
        class_likelihood_ratios(**params)