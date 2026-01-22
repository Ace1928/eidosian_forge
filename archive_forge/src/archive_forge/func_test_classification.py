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
@pytest.mark.parametrize('base_estimator', [KNeighborsClassifier(), SVC(gamma='scale', probability=True, random_state=0)])
@pytest.mark.parametrize('selection_crit', ['threshold', 'k_best'])
def test_classification(base_estimator, selection_crit):
    threshold = 0.75
    max_iter = 10
    st = SelfTrainingClassifier(base_estimator, max_iter=max_iter, threshold=threshold, criterion=selection_crit)
    st.fit(X_train, y_train_missing_labels)
    pred = st.predict(X_test)
    proba = st.predict_proba(X_test)
    st_string = SelfTrainingClassifier(base_estimator, max_iter=max_iter, criterion=selection_crit, threshold=threshold)
    st_string.fit(X_train, y_train_missing_strings)
    pred_string = st_string.predict(X_test)
    proba_string = st_string.predict_proba(X_test)
    assert_array_equal(np.vectorize(mapping.get)(pred), pred_string)
    assert_array_equal(proba, proba_string)
    assert st.termination_condition_ == st_string.termination_condition_
    labeled = y_train_missing_labels != -1
    assert_array_equal(st.labeled_iter_ == 0, labeled)
    assert_array_equal(y_train_missing_labels[labeled], st.transduction_[labeled])
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter
    assert np.max(st_string.labeled_iter_) <= st_string.n_iter_ <= max_iter
    assert st.labeled_iter_.shape == st.transduction_.shape
    assert st_string.labeled_iter_.shape == st_string.transduction_.shape