import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import (
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration(data, method, csr_container, ensemble):
    n_samples = N_SAMPLES // 2
    X, y = data
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)
    X -= X.min()
    X_train, y_train, sw_train = (X[:n_samples], y[:n_samples], sample_weight[:n_samples])
    X_test, y_test = (X[n_samples:], y[n_samples:])
    clf = MultinomialNB().fit(X_train, y_train, sample_weight=sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
    cal_clf = CalibratedClassifierCV(clf, cv=y.size + 1, ensemble=ensemble)
    with pytest.raises(ValueError):
        cal_clf.fit(X, y)
    for this_X_train, this_X_test in [(X_train, X_test), (csr_container(X_train), csr_container(X_test))]:
        cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
        cal_clf.fit(this_X_train, y_train, sample_weight=sw_train)
        prob_pos_cal_clf = cal_clf.predict_proba(this_X_test)[:, 1]
        assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(y_test, prob_pos_cal_clf)
        cal_clf.fit(this_X_train, y_train + 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)
        cal_clf.fit(this_X_train, 2 * y_train - 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)
        cal_clf.fit(this_X_train, (y_train + 1) % 2, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        if method == 'sigmoid':
            assert_array_almost_equal(prob_pos_cal_clf, 1 - prob_pos_cal_clf_relabeled)
        else:
            assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss((y_test + 1) % 2, prob_pos_cal_clf_relabeled)