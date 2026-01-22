import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.parametrize('kwargs, expected_names', [({'voting': 'soft', 'flatten_transform': True}, ['votingclassifier_lr0', 'votingclassifier_lr1', 'votingclassifier_lr2', 'votingclassifier_tree0', 'votingclassifier_tree1', 'votingclassifier_tree2']), ({'voting': 'hard'}, ['votingclassifier_lr', 'votingclassifier_tree'])])
def test_get_features_names_out_classifier(kwargs, expected_names):
    """Check get_feature_names_out for classifier for different settings."""
    X = [[1, 2], [3, 4], [5, 6], [1, 1.2]]
    y = [0, 1, 2, 0]
    voting = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=0)), ('tree', DecisionTreeClassifier(random_state=0))], **kwargs)
    voting.fit(X, y)
    X_trans = voting.transform(X)
    names_out = voting.get_feature_names_out()
    assert X_trans.shape[1] == len(expected_names)
    assert_array_equal(names_out, expected_names)