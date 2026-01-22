from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('method, numeric_only, expected_data', [('sum', True, ()), ('sum', False, ['cat_1cat_2']), ('sum', lib.no_default, ['cat_1cat_2']), ('prod', True, ()), ('prod', False, ()), ('prod', lib.no_default, ()), ('min', True, ()), ('min', False, ['cat_1']), ('min', lib.no_default, ['cat_1']), ('max', True, ()), ('max', False, ['cat_2']), ('max', lib.no_default, ['cat_2']), ('first', True, ()), ('first', False, ['cat_1']), ('first', lib.no_default, ['cat_1']), ('last', True, ()), ('last', False, ['cat_2']), ('last', lib.no_default, ['cat_2'])])
def test_series_downsample_method(method, numeric_only, expected_data):
    index = date_range('2018-01-01', periods=2, freq='D')
    expected_index = date_range('2018-12-31', periods=1, freq='YE')
    df = Series(['cat_1', 'cat_2'], index=index)
    resampled = df.resample('YE')
    kwargs = {} if numeric_only is lib.no_default else {'numeric_only': numeric_only}
    func = getattr(resampled, method)
    if numeric_only and numeric_only is not lib.no_default:
        msg = f'Cannot use numeric_only=True with SeriesGroupBy\\.{method}'
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    elif method == 'prod':
        msg = re.escape('agg function failed [how->prod,dtype->')
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    else:
        result = func(**kwargs)
        expected = Series(expected_data, index=expected_index)
        tm.assert_series_equal(result, expected)