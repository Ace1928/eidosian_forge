from statsmodels.compat.pandas import PD_LT_2_2_0
from datetime import datetime
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
def test_pandas_nodates_index():
    data = [988, 819, 964]
    dates = ['a', 'b', 'c']
    s = pd.Series(data, index=dates)
    data = [988, 819, 964]
    index = pd.to_datetime([100, 101, 102])
    s = pd.Series(data, index=index)
    actual_str = index[0].strftime('%Y-%m-%d %H:%M:%S.%f') + str(index[0].value)
    assert_equal(actual_str, '1970-01-01 00:00:00.000000100')
    with pytest.warns(ValueWarning, match='No frequency information'):
        mod = TimeSeriesModel(s)
    start, end, out_of_sample, _ = mod._get_prediction_index(0, 4)
    assert_equal(len(mod.data.predict_dates), 5)