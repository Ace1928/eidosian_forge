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
def test_calibration_prefit(csr_container):
    """Test calibration for prefitted classifiers"""
    n_samples = 50
    X, y = make_classification(n_samples=3 * n_samples, n_features=6, random_state=42)
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)
    X -= X.min()
    X_train, y_train, sw_train = (X[:n_samples], y[:n_samples], sample_weight[:n_samples])
    X_calib, y_calib, sw_calib = (X[n_samples:2 * n_samples], y[n_samples:2 * n_samples], sample_weight[n_samples:2 * n_samples])
    X_test, y_test = (X[2 * n_samples:], y[2 * n_samples:])
    clf = MultinomialNB()
    unfit_clf = CalibratedClassifierCV(clf, cv='prefit')
    with pytest.raises(NotFittedError):
        unfit_clf.fit(X_calib, y_calib)
    clf.fit(X_train, y_train, sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
    for this_X_calib, this_X_test in [(X_calib, X_test), (csr_container(X_calib), csr_container(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            cal_clf = CalibratedClassifierCV(clf, method=method, cv='prefit')
            for sw in [sw_calib, None]:
                cal_clf.fit(this_X_calib, y_calib, sample_weight=sw)
                y_prob = cal_clf.predict_proba(this_X_test)
                y_pred = cal_clf.predict(this_X_test)
                prob_pos_cal_clf = y_prob[:, 1]
                assert_array_equal(y_pred, np.array([0, 1])[np.argmax(y_prob, axis=1)])
                assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(y_test, prob_pos_cal_clf)