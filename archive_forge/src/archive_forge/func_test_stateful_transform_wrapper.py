from __future__ import print_function
import numpy as np
from patsy.state import Center, Standardize, center
from patsy.util import atleast_2d_column_default
def test_stateful_transform_wrapper():
    assert np.allclose(center([1, 2, 3]), [-1, 0, 1])
    assert np.allclose(center([1, 2, 1, 2]), [-0.5, 0.5, -0.5, 0.5])
    assert center([1.0, 2.0, 3.0]).dtype == np.dtype(float)
    assert center(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == np.dtype(np.float32)
    assert center([1, 2, 3]).dtype == np.dtype(float)
    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        s = pandas.Series([1, 2, 3], index=['a', 'b', 'c'])
        df = pandas.DataFrame([[1, 2], [2, 4], [3, 6]], columns=['x1', 'x2'], index=[10, 20, 30])
        s_c = center(s)
        assert isinstance(s_c, pandas.Series)
        assert np.array_equal(s_c.index, ['a', 'b', 'c'])
        assert np.allclose(s_c, [-1, 0, 1])
        df_c = center(df)
        assert isinstance(df_c, pandas.DataFrame)
        assert np.array_equal(df_c.index, [10, 20, 30])
        assert np.array_equal(df_c.columns, ['x1', 'x2'])
        assert np.allclose(df_c, [[-1, -2], [0, 0], [1, 2]])