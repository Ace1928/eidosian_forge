import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_linear_regression_pd_sparse_dataframe_warning():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'0': np.random.randn(10)})
    for col in range(1, 4):
        arr = np.random.randn(10)
        arr[:8] = 0
        if col != 0:
            arr = pd.arrays.SparseArray(arr, fill_value=0)
        df[str(col)] = arr
    msg = 'pandas.DataFrame with sparse columns found.'
    reg = LinearRegression()
    with pytest.warns(UserWarning, match=msg):
        reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])
    df['0'] = pd.arrays.SparseArray(df['0'], fill_value=0)
    assert hasattr(df, 'sparse')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])