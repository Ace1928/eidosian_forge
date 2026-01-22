from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouping_by_key_is_in_axis():
    df = DataFrame({'a': [1, 1, 2], 'b': [1, 1, 2], 'c': [3, 4, 5]}).set_index('a')
    gb = df.groupby([Grouper(level='a'), Grouper(key='b')], as_index=False)
    assert not gb._grouper.groupings[0].in_axis
    assert gb._grouper.groupings[1].in_axis
    msg = 'A grouping .* was excluded from the result'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.sum()
    expected = DataFrame({'b': [1, 2], 'c': [7, 5]})
    tm.assert_frame_equal(result, expected)