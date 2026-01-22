from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('mi1_list', [[['a'], range(2)], [['b'], np.arange(2.0, 4.0)], [['c'], ['A', 'B']], [['d'], pd.date_range(start='2017', end='2018', periods=2)]])
@pytest.mark.parametrize('mi2_list', [[['a'], range(2)], [['b'], np.arange(2.0, 4.0)], [['c'], ['A', 'B']], [['d'], pd.date_range(start='2017', end='2018', periods=2)]])
def test_concat_with_various_multiindex_dtypes(self, mi1_list: list, mi2_list: list):
    mi1 = MultiIndex.from_product(mi1_list)
    mi2 = MultiIndex.from_product(mi2_list)
    df1 = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
    df2 = DataFrame(np.zeros((1, len(mi2))), columns=mi2)
    if mi1_list[0] == mi2_list[0]:
        expected_mi = MultiIndex(levels=[mi1_list[0], list(mi1_list[1])], codes=[[0, 0, 0, 0], [0, 1, 0, 1]])
    else:
        expected_mi = MultiIndex(levels=[mi1_list[0] + mi2_list[0], list(mi1_list[1]) + list(mi2_list[1])], codes=[[0, 0, 1, 1], [0, 1, 2, 3]])
    expected_df = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)
    with tm.assert_produces_warning(None):
        result_df = concat((df1, df2), axis=1)
    tm.assert_frame_equal(expected_df, result_df)