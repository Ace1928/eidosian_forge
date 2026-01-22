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
def test_classification_report_dictionary_output():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
    expected_report = {'setosa': {'precision': 0.8260869565217391, 'recall': 0.7916666666666666, 'f1-score': 0.8085106382978724, 'support': 24}, 'versicolor': {'precision': 0.3333333333333333, 'recall': 0.0967741935483871, 'f1-score': 0.15000000000000002, 'support': 31}, 'virginica': {'precision': 0.4186046511627907, 'recall': 0.9, 'f1-score': 0.5714285714285715, 'support': 20}, 'macro avg': {'f1-score': 0.5099797365754813, 'precision': 0.5260083136726211, 'recall': 0.596146953405018, 'support': 75}, 'accuracy': 0.5333333333333333, 'weighted avg': {'f1-score': 0.47310435663627154, 'precision': 0.5137535108414785, 'recall': 0.5333333333333333, 'support': 75}}
    report = classification_report(y_true, y_pred, labels=np.arange(len(iris.target_names)), target_names=iris.target_names, output_dict=True)
    assert report.keys() == expected_report.keys()
    for key in expected_report:
        if key == 'accuracy':
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])
    assert isinstance(expected_report['setosa']['precision'], float)
    assert isinstance(expected_report['macro avg']['precision'], float)
    assert isinstance(expected_report['setosa']['support'], int)
    assert isinstance(expected_report['macro avg']['support'], int)