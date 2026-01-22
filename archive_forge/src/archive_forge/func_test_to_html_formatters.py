from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('df,formatters,expected', [(DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], columns=Index(['foo', None], dtype=object), index=np.arange(4)), {'__index__': lambda x: 'abcd'[x]}, 'index_formatter'), (DataFrame({'months': [datetime(2016, 1, 1), datetime(2016, 2, 2)]}), {'months': lambda x: x.strftime('%Y-%m')}, 'datetime64_monthformatter'), (DataFrame({'hod': pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')}), {'hod': lambda x: x.strftime('%H:%M')}, 'datetime64_hourformatter'), (DataFrame({'i': pd.Series([1, 2], dtype='int64'), 'f': pd.Series([1, 2], dtype='float64'), 'I': pd.Series([1, 2], dtype='Int64'), 's': pd.Series([1, 2], dtype='string'), 'b': pd.Series([True, False], dtype='boolean'), 'c': pd.Series(['a', 'b'], dtype=pd.CategoricalDtype(['a', 'b'])), 'o': pd.Series([1, '2'], dtype=object)}), [lambda x: 'formatted'] * 7, 'various_dtypes_formatted')])
def test_to_html_formatters(df, formatters, expected, datapath):
    expected = expected_html(datapath, expected)
    result = df.to_html(formatters=formatters)
    assert result == expected