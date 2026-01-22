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
def test_sanity_classification():
    base_estimator = SVC(gamma='scale', probability=True)
    base_estimator.fit(X_train[n_labeled_samples:], y_train[n_labeled_samples:])
    st = SelfTrainingClassifier(base_estimator)
    st.fit(X_train, y_train_missing_labels)
    pred1, pred2 = (base_estimator.predict(X_test), st.predict(X_test))
    assert not np.array_equal(pred1, pred2)
    score_supervised = accuracy_score(base_estimator.predict(X_test), y_test)
    score_self_training = accuracy_score(st.predict(X_test), y_test)
    assert score_self_training > score_supervised