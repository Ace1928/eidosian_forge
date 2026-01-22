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
def test_column_transformer_get_set_params_with_remainder():
    ct = ColumnTransformer([('trans1', StandardScaler(), [0])], remainder=StandardScaler())
    exp = {'n_jobs': None, 'remainder': ct.remainder, 'remainder__copy': True, 'remainder__with_mean': True, 'remainder__with_std': True, 'sparse_threshold': 0.3, 'trans1': ct.transformers[0][1], 'trans1__copy': True, 'trans1__with_mean': True, 'trans1__with_std': True, 'transformers': ct.transformers, 'transformer_weights': None, 'verbose_feature_names_out': True, 'verbose': False}
    assert ct.get_params() == exp
    ct.set_params(remainder__with_std=False)
    assert not ct.get_params()['remainder__with_std']
    ct.set_params(trans1='passthrough')
    exp = {'n_jobs': None, 'remainder': ct.remainder, 'remainder__copy': True, 'remainder__with_mean': True, 'remainder__with_std': False, 'sparse_threshold': 0.3, 'trans1': 'passthrough', 'transformers': ct.transformers, 'transformer_weights': None, 'verbose_feature_names_out': True, 'verbose': False}
    assert ct.get_params() == exp