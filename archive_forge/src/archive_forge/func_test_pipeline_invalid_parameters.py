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
def test_pipeline_invalid_parameters():
    pipeline = Pipeline([(1, 1)])
    with pytest.raises(TypeError):
        pipeline.fit([[1]], [1])
    msg = "Last step of Pipeline should implement fit or be the string 'passthrough'.*NoFit.*"
    pipeline = Pipeline([('clf', NoFit())])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])
    clf = NoTrans()
    pipe = Pipeline([('svc', clf)])
    assert pipe.get_params(deep=True) == dict(svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False))
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    repr(pipe)
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([('anova', filter1), ('svc', clf)])
    assert pipe.named_steps['anova'] is filter1
    assert pipe.named_steps['svc'] is clf
    msg = 'All intermediate steps should be transformers.*\\bNoTrans\\b.*'
    pipeline = Pipeline([('t', NoTrans()), ('svc', clf)])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    repr(pipe)
    msg = re.escape("Invalid parameter 'C' for estimator SelectKBest(). Valid parameters are: ['k', 'score_func'].")
    with pytest.raises(ValueError, match=msg):
        pipe.set_params(anova__C=0.1)
    pipe2 = clone(pipe)
    assert pipe.named_steps['svc'] is not pipe2.named_steps['svc']
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)
    for x in pipe.get_params(deep=False):
        params.pop(x)
    for x in pipe2.get_params(deep=False):
        params2.pop(x)
    params.pop('svc')
    params.pop('anova')
    params2.pop('svc')
    params2.pop('anova')
    assert params == params2