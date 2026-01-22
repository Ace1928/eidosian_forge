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
def test_step_name_validation():
    error_message_1 = "Estimator names must not contain __: got \\['a__q'\\]"
    error_message_2 = "Names provided are not unique: \\['a', 'a'\\]"
    error_message_3 = "Estimator names conflict with constructor arguments: \\['%s'\\]"
    bad_steps1 = [('a__q', Mult(2)), ('b', Mult(3))]
    bad_steps2 = [('a', Mult(2)), ('a', Mult(3))]
    for cls, param in [(Pipeline, 'steps'), (FeatureUnion, 'transformer_list')]:
        bad_steps3 = [('a', Mult(2)), (param, Mult(3))]
        for bad_steps, message in [(bad_steps1, error_message_1), (bad_steps2, error_message_2), (bad_steps3, error_message_3 % param)]:
            with pytest.raises(ValueError, match=message):
                cls(**{param: bad_steps}).fit([[1]], [1])
            est = cls(**{param: [('a', Mult(1))]})
            setattr(est, param, bad_steps)
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])
            with pytest.raises(ValueError, match=message):
                est.fit_transform([[1]], [1])
            est = cls(**{param: [('a', Mult(1))]})
            est.set_params(**{param: bad_steps})
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])
            with pytest.raises(ValueError, match=message):
                est.fit_transform([[1]], [1])