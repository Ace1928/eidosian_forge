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
@pytest.mark.parametrize('trans_1, expected_verbose_names, expected_non_verbose_names', [(PandasOutTransformer(offset=2.0), ['trans_0__feat1', 'trans_1__feat0'], ['feat1', 'feat0']), ('drop', ['trans_0__feat1'], ['feat1']), ('passthrough', ['trans_0__feat1', 'trans_1__feat0'], ['feat1', 'feat0'])])
def test_transformers_with_pandas_out_but_not_feature_names_out(trans_1, expected_verbose_names, expected_non_verbose_names):
    """Check that set_config(transform="pandas") is compatible with more transformers.

    Specifically, if transformers returns a DataFrame, but does not define
    `get_feature_names_out`.
    """
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'feat0': [1.0, 2.0, 3.0], 'feat1': [2.0, 3.0, 4.0]})
    ct = ColumnTransformer([('trans_0', PandasOutTransformer(offset=3.0), ['feat1']), ('trans_1', trans_1, ['feat0'])])
    X_trans_np = ct.fit_transform(X_df)
    assert isinstance(X_trans_np, np.ndarray)
    with pytest.raises(AttributeError, match='not provide get_feature_names_out'):
        ct.get_feature_names_out()
    ct.set_output(transform='pandas')
    X_trans_df0 = ct.fit_transform(X_df)
    assert_array_equal(X_trans_df0.columns, expected_verbose_names)
    ct.set_params(verbose_feature_names_out=False)
    X_trans_df1 = ct.fit_transform(X_df)
    assert_array_equal(X_trans_df1.columns, expected_non_verbose_names)