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
def test_base_chain_crossval_fit_and_predict():
    X, Y = generate_multilabel_dataset_with_correlations()
    for chain in [ClassifierChain(LogisticRegression()), RegressorChain(Ridge())]:
        chain.fit(X, Y)
        chain_cv = clone(chain).set_params(cv=3)
        chain_cv.fit(X, Y)
        Y_pred_cv = chain_cv.predict(X)
        Y_pred = chain.predict(X)
        assert Y_pred_cv.shape == Y_pred.shape
        assert not np.all(Y_pred == Y_pred_cv)
        if isinstance(chain, ClassifierChain):
            assert jaccard_score(Y, Y_pred_cv, average='samples') > 0.4
        else:
            assert mean_squared_error(Y, Y_pred_cv) < 0.25