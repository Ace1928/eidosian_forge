from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
@pytest.mark.parametrize('Est', (HalvingGridSearchCV, HalvingRandomSearchCV))
def test_groups_support(Est):
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 50)
    clf = LinearSVC(dual='auto', random_state=0)
    grid = {'C': [1]}
    group_cvs = [LeaveOneGroupOut(), LeavePGroupsOut(2), GroupKFold(n_splits=3), GroupShuffleSplit(random_state=0)]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = Est(clf, grid, cv=cv, random_state=0)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)
    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit(random_state=0)]
    for cv in non_group_cvs:
        gs = Est(clf, grid, cv=cv)
        gs.fit(X, y)