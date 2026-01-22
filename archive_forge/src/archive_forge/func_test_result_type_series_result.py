from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_result_type_series_result(int_frame_const_col, engine, request):
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba Series constructor only support ndarrays not list data')
        request.node.add_marker(mark)
    df = int_frame_const_col
    result = df.apply(lambda x: Series([1, 2, 3], index=x.index), axis=1, engine=engine)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)