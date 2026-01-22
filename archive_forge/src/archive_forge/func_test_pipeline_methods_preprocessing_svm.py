import itertools
import re
import shutil
import time
from tempfile import mkdtemp
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import check_is_fitted
def test_pipeline_methods_preprocessing_svm():
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver='randomized', whiten=True)
    clf = SVC(probability=True, random_state=0, decision_function_shape='ovr')
    for preprocessing in [scaler, pca]:
        pipe = Pipeline([('preprocess', preprocessing), ('svc', clf)])
        pipe.fit(X, y)
        predict = pipe.predict(X)
        assert predict.shape == (n_samples,)
        proba = pipe.predict_proba(X)
        assert proba.shape == (n_samples, n_classes)
        log_proba = pipe.predict_log_proba(X)
        assert log_proba.shape == (n_samples, n_classes)
        decision_function = pipe.decision_function(X)
        assert decision_function.shape == (n_samples, n_classes)
        pipe.score(X, y)