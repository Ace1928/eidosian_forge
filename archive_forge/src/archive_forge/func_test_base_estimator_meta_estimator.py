from math import ceil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
def test_base_estimator_meta_estimator():
    base_estimator = StackingClassifier(estimators=[('svc_1', SVC(probability=True)), ('svc_2', SVC(probability=True))], final_estimator=SVC(probability=True), cv=2)
    assert hasattr(base_estimator, 'predict_proba')
    clf = SelfTrainingClassifier(base_estimator=base_estimator)
    clf.fit(X_train, y_train_missing_labels)
    clf.predict_proba(X_test)
    base_estimator = StackingClassifier(estimators=[('svc_1', SVC(probability=False)), ('svc_2', SVC(probability=False))], final_estimator=SVC(probability=False), cv=2)
    assert not hasattr(base_estimator, 'predict_proba')
    clf = SelfTrainingClassifier(base_estimator=base_estimator)
    with pytest.raises(AttributeError):
        clf.fit(X_train, y_train_missing_labels)