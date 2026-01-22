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
def test_k_best():
    st = SelfTrainingClassifier(KNeighborsClassifier(n_neighbors=1), criterion='k_best', k_best=10, max_iter=None)
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]
    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)
    assert st.n_iter_ == n_expected_iter
    assert np.sum(st.labeled_iter_ == 0) == 1
    for i in range(1, n_expected_iter):
        assert np.sum(st.labeled_iter_ == i) == 10
    assert np.sum(st.labeled_iter_ == n_expected_iter) == (n_samples - 1) % 10
    assert st.termination_condition_ == 'all_labeled'