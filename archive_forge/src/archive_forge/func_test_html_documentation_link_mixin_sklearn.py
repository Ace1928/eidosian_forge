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
@pytest.mark.parametrize('mock_version', ['1.3.0.dev0', '1.3.0'])
def test_html_documentation_link_mixin_sklearn(mock_version):
    """Check the behaviour of the `_HTMLDocumentationLinkMixin` class for scikit-learn
    default.
    """
    with patch('sklearn.utils._estimator_html_repr.__version__', mock_version):
        mixin = _HTMLDocumentationLinkMixin()
        assert mixin._doc_link_module == 'sklearn'
        sklearn_version = parse_version(mock_version)
        if sklearn_version.dev is None:
            version = f'{sklearn_version.major}.{sklearn_version.minor}'
        else:
            version = 'dev'
        assert mixin._doc_link_template == f'https://scikit-learn.org/{version}/modules/generated/{{estimator_module}}.{{estimator_name}}.html'
        assert mixin._get_doc_link() == f'https://scikit-learn.org/{version}/modules/generated/sklearn.utils._HTMLDocumentationLinkMixin.html'