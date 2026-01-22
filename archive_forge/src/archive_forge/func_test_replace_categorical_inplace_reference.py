import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('to_replace', [1, [1]])
@pytest.mark.parametrize('val', [1, 1.5])
def test_replace_categorical_inplace_reference(using_copy_on_write, val, to_replace):
    df = DataFrame({'a': Categorical([1, 2, 3])})
    df_orig = df.copy()
    arr_a = get_array(df, 'a')
    view = df[:]
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    warn = FutureWarning if val == 1.5 else None
    with tm.assert_produces_warning(warn, match=msg):
        df.replace(to_replace=to_replace, value=val, inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a').codes, arr_a.codes)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a').codes, arr_a.codes)