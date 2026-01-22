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
def test_sample_weight(global_random_seed):
    """Tests sample_weight parameter of VotingClassifier"""
    clf1 = LogisticRegression(random_state=global_random_seed)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    clf3 = SVC(probability=True, random_state=global_random_seed)
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft').fit(X_scaled, y, sample_weight=np.ones((len(y),)))
    eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft').fit(X_scaled, y)
    assert_array_equal(eclf1.predict(X_scaled), eclf2.predict(X_scaled))
    assert_array_almost_equal(eclf1.predict_proba(X_scaled), eclf2.predict_proba(X_scaled))
    sample_weight = np.random.RandomState(global_random_seed).uniform(size=(len(y),))
    eclf3 = VotingClassifier(estimators=[('lr', clf1)], voting='soft')
    eclf3.fit(X_scaled, y, sample_weight)
    clf1.fit(X_scaled, y, sample_weight)
    assert_array_equal(eclf3.predict(X_scaled), clf1.predict(X_scaled))
    assert_array_almost_equal(eclf3.predict_proba(X_scaled), clf1.predict_proba(X_scaled))
    clf4 = KNeighborsClassifier()
    eclf3 = VotingClassifier(estimators=[('lr', clf1), ('svc', clf3), ('knn', clf4)], voting='soft')
    msg = 'Underlying estimator KNeighborsClassifier does not support sample weights.'
    with pytest.raises(TypeError, match=msg):
        eclf3.fit(X_scaled, y, sample_weight)

    class ClassifierErrorFit(ClassifierMixin, BaseEstimator):

        def fit(self, X_scaled, y, sample_weight):
            raise TypeError('Error unrelated to sample_weight.')
    clf = ClassifierErrorFit()
    with pytest.raises(TypeError, match='Error unrelated to sample_weight'):
        clf.fit(X_scaled, y, sample_weight=sample_weight)