import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_column_transformer_get_set_params():
    ct = ColumnTransformer([('trans1', StandardScaler(), [0]), ('trans2', StandardScaler(), [1])])
    exp = {'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'trans1': ct.transformers[0][1], 'trans1__copy': True, 'trans1__with_mean': True, 'trans1__with_std': True, 'trans2': ct.transformers[1][1], 'trans2__copy': True, 'trans2__with_mean': True, 'trans2__with_std': True, 'transformers': ct.transformers, 'transformer_weights': None, 'verbose_feature_names_out': True, 'verbose': False}
    assert ct.get_params() == exp
    ct.set_params(trans1__with_mean=False)
    assert not ct.get_params()['trans1__with_mean']
    ct.set_params(trans1='passthrough')
    exp = {'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'trans1': 'passthrough', 'trans2': ct.transformers[1][1], 'trans2__copy': True, 'trans2__with_mean': True, 'trans2__with_std': True, 'transformers': ct.transformers, 'transformer_weights': None, 'verbose_feature_names_out': True, 'verbose': False}
    assert ct.get_params() == exp