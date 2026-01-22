import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('copy', [True, None, False])
def test_concat_copy_keyword(using_copy_on_write, copy):
    df = DataFrame({'a': [1, 2]})
    df2 = DataFrame({'b': [1.5, 2.5]})
    result = concat([df, df2], axis=1, copy=copy)
    if using_copy_on_write or copy is False:
        assert np.shares_memory(get_array(df, 'a'), get_array(result, 'a'))
        assert np.shares_memory(get_array(df2, 'b'), get_array(result, 'b'))
    else:
        assert not np.shares_memory(get_array(df, 'a'), get_array(result, 'a'))
        assert not np.shares_memory(get_array(df2, 'b'), get_array(result, 'b'))