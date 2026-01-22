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
@pytest.mark.parametrize('selector', [[1], lambda x: [1], ['col2'], lambda x: ['col2'], [False, True], lambda x: [False, True]])
def test_feature_names_out_pandas(selector):
    """Checks name when selecting only the second column"""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'col1': ['a', 'a', 'b'], 'col2': ['z', 'z', 'z']})
    ct = ColumnTransformer([('ohe', OneHotEncoder(), selector)])
    ct.fit(df)
    assert_array_equal(ct.get_feature_names_out(), ['ohe__col2_z'])