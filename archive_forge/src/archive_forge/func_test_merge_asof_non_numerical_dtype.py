import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('kwargs', [{'on': 'x'}, {'left_index': True, 'right_index': True}])
@pytest.mark.parametrize('data', [['2019-06-01 00:09:12', '2019-06-01 00:10:29'], [1.0, '2019-06-01 00:10:29']])
def test_merge_asof_non_numerical_dtype(kwargs, data, infer_string):
    with option_context('future.infer_string', infer_string):
        left = pd.DataFrame({'x': data}, index=data)
        right = pd.DataFrame({'x': data}, index=data)
        with pytest.raises(MergeError, match='Incompatible merge dtype, .*, both sides must have numeric dtype'):
            merge_asof(left, right, **kwargs)