import html
import locale
import re
from contextlib import closing
from io import StringIO
from unittest.mock import patch
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._estimator_html_repr import (
from sklearn.utils.fixes import parse_version
@pytest.mark.parametrize('estimator', [LogisticRegression(), make_pipeline(StandardScaler(), LogisticRegression()), make_pipeline(make_column_transformer((StandardScaler(), slice(0, 3))), LogisticRegression())])
def test_estimator_html_repr_fitted_icon(estimator):
    """Check that we are showing the fitted status icon only once."""
    pattern = '<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span>'
    assert estimator_html_repr(estimator).count(pattern) == 1
    X, y = load_iris(return_X_y=True)
    estimator.fit(X, y)
    pattern = '<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span>'
    assert estimator_html_repr(estimator).count(pattern) == 1