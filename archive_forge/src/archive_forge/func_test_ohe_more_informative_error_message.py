import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_more_informative_error_message():
    """Raise informative error message when pandas output and sparse_output=True."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['z', 'b', 'b']}, columns=['a', 'b'])
    ohe = OneHotEncoder(sparse_output=True)
    ohe.set_output(transform='pandas')
    msg = 'Pandas output does not support sparse data. Set sparse_output=False to output pandas dataframes or disable Pandas output'
    with pytest.raises(ValueError, match=msg):
        ohe.fit_transform(df)
    ohe.fit(df)
    with pytest.raises(ValueError, match=msg):
        ohe.transform(df)