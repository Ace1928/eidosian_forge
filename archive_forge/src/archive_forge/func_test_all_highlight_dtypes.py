import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('f,kwargs', [('highlight_min', {'axis': 1, 'subset': IndexSlice[1, :]}), ('highlight_max', {'axis': 0, 'subset': [0]}), ('highlight_quantile', {'axis': None, 'q_left': 0.6, 'q_right': 0.8}), ('highlight_between', {'subset': [0]})])
@pytest.mark.parametrize('df', [DataFrame([[0, 10], [20, 30]], dtype=int), DataFrame([[0, 10], [20, 30]], dtype=float), DataFrame([[0, 10], [20, 30]], dtype='datetime64[ns]'), DataFrame([[0, 10], [20, 30]], dtype=str), DataFrame([[0, 10], [20, 30]], dtype='timedelta64[ns]')])
def test_all_highlight_dtypes(f, kwargs, df):
    if f == 'highlight_quantile' and isinstance(df.iloc[0, 0], str):
        return None
    if f == 'highlight_between':
        kwargs['left'] = df.iloc[1, 0]
    expected = {(1, 0): [('background-color', 'yellow')]}
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected