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
def test_k_best_selects_best():
    svc = SVC(gamma='scale', probability=True, random_state=0)
    st = SelfTrainingClassifier(svc, criterion='k_best', max_iter=1, k_best=10)
    has_label = y_train_missing_labels != -1
    st.fit(X_train, y_train_missing_labels)
    got_label = ~has_label & (st.transduction_ != -1)
    svc.fit(X_train[has_label], y_train_missing_labels[has_label])
    pred = svc.predict_proba(X_train[~has_label])
    max_proba = np.max(pred, axis=1)
    most_confident_svc = X_train[~has_label][np.argsort(max_proba)[-10:]]
    added_by_st = X_train[np.where(got_label)].tolist()
    for row in most_confident_svc.tolist():
        assert row in added_by_st