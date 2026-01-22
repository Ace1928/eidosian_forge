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
@pytest.mark.parametrize('empty_col', [[], np.array([], dtype=int), lambda x: []], ids=['list', 'array', 'callable'])
def test_feature_names_empty_columns(empty_col):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'col1': ['a', 'a', 'b'], 'col2': ['z', 'z', 'z']})
    ct = ColumnTransformer(transformers=[('ohe', OneHotEncoder(), ['col1', 'col2']), ('empty_features', OneHotEncoder(), empty_col)])
    ct.fit(df)
    assert_array_equal(ct.get_feature_names_out(), ['ohe__col1_a', 'ohe__col1_b', 'ohe__col2_z'])