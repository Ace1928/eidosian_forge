from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_col,expected', [([0], DataFrame({'b': [np.nan], 'c': [1], 'd': [5]}, index=Index([0], name='a'))), ([0, 2], DataFrame({'b': [np.nan], 'd': [5]}, index=MultiIndex.from_tuples([(0, 1)], names=['a', 'c']))), (['a', 'c'], DataFrame({'b': [np.nan], 'd': [5]}, index=MultiIndex.from_tuples([(0, 1)], names=['a', 'c'])))])
def test_na_value_dict_multi_index(all_parsers, index_col, expected):
    data = 'a,b,c,d\n0,NA,1,5\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    tm.assert_frame_equal(result, expected)