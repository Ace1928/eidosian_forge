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
def test_transform_pd_na():
    """Check behavior when a tranformer's output contains pandas.NA

    It should emit a warning unless the output config is set to 'pandas'.
    """
    pd = pytest.importorskip('pandas')
    if not hasattr(pd, 'Float64Dtype'):
        pytest.skip('The issue with pd.NA tested here does not happen in old versions that do not have the extension dtypes')
    df = pd.DataFrame({'a': [1.5, None]})
    ct = make_column_transformer(('passthrough', ['a']))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ct.fit_transform(df)
    df = df.convert_dtypes()
    with pytest.warns(FutureWarning, match="set_output\\(transform='pandas'\\)"):
        ct.fit_transform(df)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ct.set_output(transform='pandas')
        ct.fit_transform(df)
    ct.set_output(transform='default')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ct.fit_transform(df.fillna(-1.0))