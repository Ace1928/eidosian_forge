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
@pytest.mark.parametrize('remainder', ['passthrough', StandardScaler()])
def test_sk_visual_block_remainder_fitted_pandas(remainder):
    pd = pytest.importorskip('pandas')
    ohe = OneHotEncoder()
    ct = ColumnTransformer(transformers=[('ohe', ohe, ['col1', 'col2'])], remainder=remainder)
    df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['z', 'z', 'z'], 'col3': [1, 2, 3], 'col4': [3, 4, 5]})
    ct.fit(df)
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ('ohe', 'remainder')
    assert visual_block.name_details == (['col1', 'col2'], ['col3', 'col4'])
    assert visual_block.estimators == (ohe, remainder)