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
def test_make_column_transformer_kwargs():
    scaler = StandardScaler()
    norm = Normalizer()
    ct = make_column_transformer((scaler, 'first'), (norm, ['second']), n_jobs=3, remainder='drop', sparse_threshold=0.5)
    assert ct.transformers == make_column_transformer((scaler, 'first'), (norm, ['second'])).transformers
    assert ct.n_jobs == 3
    assert ct.remainder == 'drop'
    assert ct.sparse_threshold == 0.5
    msg = re.escape("make_column_transformer() got an unexpected keyword argument 'transformer_weights'")
    with pytest.raises(TypeError, match=msg):
        make_column_transformer((scaler, 'first'), (norm, ['second']), transformer_weights={'pca': 10, 'Transf': 1})