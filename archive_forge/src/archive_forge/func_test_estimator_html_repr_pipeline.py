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
def test_estimator_html_repr_pipeline():
    num_trans = Pipeline(steps=[('pass', 'passthrough'), ('imputer', SimpleImputer(strategy='median'))])
    cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', missing_values='empty')), ('one-hot', OneHotEncoder(drop='first'))])
    preprocess = ColumnTransformer([('num', num_trans, ['a', 'b', 'c', 'd', 'e']), ('cat', cat_trans, [0, 1, 2, 3])])
    feat_u = FeatureUnion([('pca', PCA(n_components=1)), ('tsvd', Pipeline([('first', TruncatedSVD(n_components=3)), ('select', SelectPercentile())]))])
    clf = VotingClassifier([('lr', LogisticRegression(solver='lbfgs', random_state=1)), ('mlp', MLPClassifier(alpha=0.001))])
    pipe = Pipeline([('preprocessor', preprocess), ('feat_u', feat_u), ('classifier', clf)])
    html_output = estimator_html_repr(pipe)
    assert html.escape(str(pipe)) in html_output
    for _, est in pipe.steps:
        assert '<div class="sk-toggleable__content "><pre>' + html.escape(str(est)) in html_output
    with config_context(print_changed_only=True):
        assert html.escape(str(num_trans['pass'])) in html_output
        assert 'passthrough</label>' in html_output
        assert html.escape(str(num_trans['imputer'])) in html_output
        for _, _, cols in preprocess.transformers:
            assert f'<pre>{html.escape(str(cols))}</pre>' in html_output
        for name, _ in feat_u.transformer_list:
            assert f'<label>{html.escape(name)}</label>' in html_output
        pca = feat_u.transformer_list[0][1]
        assert f'<pre>{html.escape(str(pca))}</pre>' in html_output
        tsvd = feat_u.transformer_list[1][1]
        first = tsvd['first']
        select = tsvd['select']
        assert f'<pre>{html.escape(str(first))}</pre>' in html_output
        assert f'<pre>{html.escape(str(select))}</pre>' in html_output
        for name, est in clf.estimators:
            assert f'<label>{html.escape(name)}</label>' in html_output
            assert f'<pre>{html.escape(str(est))}</pre>' in html_output
    assert 'prefers-color-scheme' in html_output