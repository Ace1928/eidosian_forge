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
def test_verbose_k_best(capsys):
    st = SelfTrainingClassifier(KNeighborsClassifier(n_neighbors=1), criterion='k_best', k_best=10, verbose=True, max_iter=None)
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]
    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)
    captured = capsys.readouterr()
    msg = 'End of iteration {}, added {} new labels.'
    for i in range(1, n_expected_iter):
        assert msg.format(i, 10) in captured.out
    assert msg.format(n_expected_iter, (n_samples - 1) % 10) in captured.out