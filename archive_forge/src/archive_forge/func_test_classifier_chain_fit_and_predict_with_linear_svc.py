import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_classifier_chain_fit_and_predict_with_linear_svc():
    X, Y = generate_multilabel_dataset_with_correlations()
    classifier_chain = ClassifierChain(LinearSVC(dual='auto'))
    classifier_chain.fit(X, Y)
    Y_pred = classifier_chain.predict(X)
    assert Y_pred.shape == Y.shape
    Y_decision = classifier_chain.decision_function(X)
    Y_binary = Y_decision >= 0
    assert_array_equal(Y_binary, Y_pred)
    assert not hasattr(classifier_chain, 'predict_proba')