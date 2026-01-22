import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3', [((('y', 'A'), 'max'), (('y', 'A'), np.mean), (('y', 'B'), 'mean'), [1, 3], [0.5, 2.5], [5.5, 7.5]), ((('y', 'A'), lambda x: max(x)), (('y', 'A'), lambda x: 1), (('y', 'B'), np.mean), [1, 3], [1, 1], [5.5, 7.5]), (pd.NamedAgg(('y', 'A'), 'max'), pd.NamedAgg(('y', 'B'), np.mean), pd.NamedAgg(('y', 'A'), lambda x: 1), [1, 3], [5.5, 7.5], [1, 1])])
def test_agg_relabel_multiindex_column(agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3):
    df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
    df.columns = MultiIndex.from_tuples([('x', 'group'), ('y', 'A'), ('y', 'B')])
    idx = Index(['a', 'b'], name=('x', 'group'))
    result = df.groupby(('x', 'group')).agg(a_max=(('y', 'A'), 'max'))
    expected = DataFrame({'a_max': [1, 3]}, index=idx)
    tm.assert_frame_equal(result, expected)
    msg = 'is currently using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(('x', 'group')).agg(col_1=agg_col1, col_2=agg_col2, col_3=agg_col3)
    expected = DataFrame({'col_1': agg_result1, 'col_2': agg_result2, 'col_3': agg_result3}, index=idx)
    tm.assert_frame_equal(result, expected)