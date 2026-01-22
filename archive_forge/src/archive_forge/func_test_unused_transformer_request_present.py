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
@pytest.mark.usefixtures('enable_slep006')
def test_unused_transformer_request_present():
    ct = ColumnTransformer([('trans', ConsumingTransformer().set_fit_request(metadata=True).set_transform_request(metadata=True), lambda X: [])])
    router = ct.get_metadata_routing()
    assert router.consumes('fit', ['metadata']) == set(['metadata'])