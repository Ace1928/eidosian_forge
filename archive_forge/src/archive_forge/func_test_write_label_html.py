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
@pytest.mark.parametrize('checked', [True, False])
def test_write_label_html(checked):
    name = 'LogisticRegression'
    tool_tip = 'hello-world'
    with closing(StringIO()) as out:
        _write_label_html(out, name, tool_tip, checked=checked)
        html_label = out.getvalue()
        p = '<label for="sk-estimator-id-[0-9]*" class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">LogisticRegression'
        re_compiled = re.compile(p)
        assert re_compiled.search(html_label)
        assert html_label.startswith('<div class="sk-label-container">')
        assert '<pre>hello-world</pre>' in html_label
        if checked:
            assert 'checked>' in html_label