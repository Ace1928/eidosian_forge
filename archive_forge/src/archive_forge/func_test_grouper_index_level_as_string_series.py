import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('levels', ['inner', 'outer', 'B', ['inner'], ['outer'], ['B'], ['inner', 'outer'], ['outer', 'inner'], ['inner', 'outer', 'B'], ['B', 'outer', 'inner']])
def test_grouper_index_level_as_string_series(series, levels):
    if isinstance(levels, list):
        groupers = [pd.Grouper(level=lv) for lv in levels]
    else:
        groupers = pd.Grouper(level=levels)
    expected = series.groupby(groupers).mean()
    result = series.groupby(levels).mean()
    tm.assert_series_equal(result, expected)