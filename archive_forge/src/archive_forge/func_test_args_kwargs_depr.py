from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('method, raises', [('sum', True), ('prod', True), ('min', True), ('max', True), ('first', False), ('last', False), ('median', False), ('mean', True), ('std', True), ('var', True), ('sem', False), ('ohlc', False), ('nunique', False)])
def test_args_kwargs_depr(method, raises):
    index = date_range('20180101', periods=3, freq='h')
    df = Series([2, 4, 6], index=index)
    resampled = df.resample('30min')
    args = ()
    func = getattr(resampled, method)
    error_msg = 'numpy operations are not valid with resample.'
    error_msg_type = 'too many arguments passed in'
    warn_msg = f'Passing additional args to DatetimeIndexResampler.{method}'
    if raises:
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            with pytest.raises(UnsupportedFunctionCall, match=error_msg):
                func(*args, 1, 2, 3, 4)
    else:
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            with pytest.raises(TypeError, match=error_msg_type):
                func(*args, 1, 2, 3, 4)