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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_base_chain_fit_and_predict_with_sparse_data_and_cv(csr_container):
    X, Y = generate_multilabel_dataset_with_correlations()
    X_sparse = csr_container(X)
    base_chains = [ClassifierChain(LogisticRegression(), cv=3), RegressorChain(Ridge(), cv=3)]
    for chain in base_chains:
        chain.fit(X_sparse, Y)
        Y_pred = chain.predict(X_sparse)
        assert Y_pred.shape == Y.shape