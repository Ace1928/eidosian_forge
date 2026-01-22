from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_result_type_broadcast(int_frame_const_col, request, engine):
    if engine == 'numba':
        mark = pytest.mark.xfail(reason="numba engine doesn't support list return")
        request.node.add_marker(mark)
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type='broadcast', engine=engine)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)