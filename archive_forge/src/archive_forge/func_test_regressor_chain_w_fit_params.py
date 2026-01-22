import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_regressor_chain_w_fit_params():
    rng = np.random.RandomState(0)
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    weight = rng.rand(y.shape[0])

    class MySGD(SGDRegressor):

        def fit(self, X, y, **fit_params):
            self.sample_weight_ = fit_params['sample_weight']
            super().fit(X, y, **fit_params)
    model = RegressorChain(MySGD())
    fit_param = {'sample_weight': weight}
    model.fit(X, y, **fit_param)
    for est in model.estimators_:
        assert est.sample_weight_ is weight