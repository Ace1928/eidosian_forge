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
def test_feature_union_feature_names():
    word_vect = CountVectorizer(analyzer='word')
    char_vect = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3))
    ft = FeatureUnion([('chars', char_vect), ('words', word_vect)])
    ft.fit(JUNK_FOOD_DOCS)
    feature_names = ft.get_feature_names_out()
    for feat in feature_names:
        assert 'chars__' in feat or 'words__' in feat
    assert len(feature_names) == 35
    ft = FeatureUnion([('tr1', Transf())]).fit([[1]])
    msg = re.escape('Transformer tr1 (type Transf) does not provide get_feature_names_out')
    with pytest.raises(AttributeError, match=msg):
        ft.get_feature_names_out()