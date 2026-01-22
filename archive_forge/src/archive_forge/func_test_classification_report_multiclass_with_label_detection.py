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
def test_classification_report_multiclass_with_label_detection():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
    expected_report = '              precision    recall  f1-score   support\n\n           0       0.83      0.79      0.81        24\n           1       0.33      0.10      0.15        31\n           2       0.42      0.90      0.57        20\n\n    accuracy                           0.53        75\n   macro avg       0.53      0.60      0.51        75\nweighted avg       0.51      0.53      0.47        75\n'
    report = classification_report(y_true, y_pred)
    assert report == expected_report