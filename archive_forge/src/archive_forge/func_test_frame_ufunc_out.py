from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('ufunc', _BASE_UFUNCS)
def test_frame_ufunc_out(ufunc):
    npfunc = getattr(np, ufunc)
    dafunc = getattr(da, ufunc)
    input_matrix = np.random.randint(1, 100, size=(20, 2))
    df = pd.DataFrame(input_matrix, columns=['A', 'B'])
    ddf = dd.from_pandas(df, 3)
    df_out = pd.DataFrame(np.random.randint(1, 100, size=(20, 2)), columns=['Y', 'Z'])
    ddf_out_np = dd.from_pandas(df_out, 3)
    ddf_out_da = dd.from_pandas(df_out, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        npfunc(ddf, out=ddf_out_np)
        dafunc(ddf, out=ddf_out_da)
        assert_eq(ddf_out_np, ddf_out_da)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = pd.DataFrame(npfunc(input_matrix), columns=['A', 'B'])
        assert_eq(ddf_out_np, expected)