import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_pandas_friendly_reshape():
    import pytest
    global have_pandas
    assert np.allclose(pandas_friendly_reshape(np.arange(10).reshape(5, 2), (2, 5)), np.arange(10).reshape(2, 5))
    if have_pandas:
        df = pandas.DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
        noop = pandas_friendly_reshape(df, (3, 1))
        assert isinstance(noop, pandas.DataFrame)
        assert np.array_equal(noop.index, ['a', 'b', 'c'])
        assert np.array_equal(noop.columns, ['x'])
        squozen = pandas_friendly_reshape(df, (3,))
        assert isinstance(squozen, pandas.Series)
        assert np.array_equal(squozen.index, ['a', 'b', 'c'])
        assert squozen.name == 'x'
        pytest.raises(ValueError, pandas_friendly_reshape, df, (4,))
        pytest.raises(ValueError, pandas_friendly_reshape, df, (1, 3))
        pytest.raises(ValueError, pandas_friendly_reshape, df, (3, 3))
        had_pandas = have_pandas
        try:
            have_pandas = False
            pytest.raises(AttributeError, pandas_friendly_reshape, df, (3,))
        finally:
            have_pandas = had_pandas