import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('key_strs,groupers', [('inner', pd.Grouper(level='inner')), (['inner'], [pd.Grouper(level='inner')]), (['B', 'inner'], ['B', pd.Grouper(level='inner')]), (['inner', 'B'], [pd.Grouper(level='inner'), 'B'])])
def test_grouper_index_level_as_string(frame, key_strs, groupers):
    if 'B' not in key_strs or 'outer' in frame.columns:
        result = frame.groupby(key_strs).mean(numeric_only=True)
        expected = frame.groupby(groupers).mean(numeric_only=True)
    else:
        result = frame.groupby(key_strs).mean()
        expected = frame.groupby(groupers).mean()
    tm.assert_frame_equal(result, expected)