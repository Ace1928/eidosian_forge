import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('handle_unknown', ['infrequent_if_exist', 'ignore'])
@pytest.mark.parametrize('pd_nan_type', ['pd.NA', 'np.nan'])
def test_ohe_missing_value_support_pandas_categorical(pd_nan_type, handle_unknown):
    pd = pytest.importorskip('pandas')
    pd_missing_value = pd.NA if pd_nan_type == 'pd.NA' else np.nan
    df = pd.DataFrame({'col1': pd.Series(['c', 'a', pd_missing_value, 'b', 'a'], dtype='category')})
    expected_df_trans = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    ohe = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    df_trans = ohe.fit_transform(df)
    assert_allclose(expected_df_trans, df_trans)
    assert len(ohe.categories_) == 1
    assert_array_equal(ohe.categories_[0][:-1], ['a', 'b', 'c'])
    assert np.isnan(ohe.categories_[0][-1])