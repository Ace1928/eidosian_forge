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
def test_ovo_classifier_duck_typing_meta():
    ovo = OneVsOneClassifier(LinearSVC(penalty='l1'))
    html_output = estimator_html_repr(ovo)
    with config_context(print_changed_only=True):
        assert f'<pre>{html.escape(str(ovo.estimator))}' in html_output
        p = '<label for="sk-estimator-id-[0-9]*" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;LinearSVC'
        re_compiled = re.compile(p)
        assert re_compiled.search(html_output)
    assert f'<pre>{html.escape(str(ovo))}' in html_output