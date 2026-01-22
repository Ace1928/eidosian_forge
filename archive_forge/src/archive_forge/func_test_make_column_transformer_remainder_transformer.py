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
def test_make_column_transformer_remainder_transformer():
    scaler = StandardScaler()
    norm = Normalizer()
    remainder = StandardScaler()
    ct = make_column_transformer((scaler, 'first'), (norm, ['second']), remainder=remainder)
    assert ct.remainder == remainder