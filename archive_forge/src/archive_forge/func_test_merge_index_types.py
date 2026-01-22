from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('index', [Index([1, 2], dtype=dtyp, name='index_col') for dtyp in tm.ALL_REAL_NUMPY_DTYPES] + [CategoricalIndex(['A', 'B'], categories=['A', 'B'], name='index_col'), RangeIndex(start=0, stop=2, name='index_col'), DatetimeIndex(['2018-01-01', '2018-01-02'], name='index_col')], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
def test_merge_index_types(index):
    left = DataFrame({'left_data': [1, 2]}, index=index)
    right = DataFrame({'right_data': [1.0, 2.0]}, index=index)
    result = left.merge(right, on=['index_col'])
    expected = DataFrame({'left_data': [1, 2], 'right_data': [1.0, 2.0]}, index=index)
    tm.assert_frame_equal(result, expected)