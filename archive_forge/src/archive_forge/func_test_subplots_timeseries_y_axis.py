import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('col', ['numeric', 'timedelta', 'datetime_no_tz', 'datetime_all_tz'])
def test_subplots_timeseries_y_axis(self, col):
    data = {'numeric': np.array([1, 2, 5]), 'timedelta': [pd.Timedelta(-10, unit='s'), pd.Timedelta(10, unit='m'), pd.Timedelta(10, unit='h')], 'datetime_no_tz': [pd.to_datetime('2017-08-01 00:00:00'), pd.to_datetime('2017-08-01 02:00:00'), pd.to_datetime('2017-08-02 00:00:00')], 'datetime_all_tz': [pd.to_datetime('2017-08-01 00:00:00', utc=True), pd.to_datetime('2017-08-01 02:00:00', utc=True), pd.to_datetime('2017-08-02 00:00:00', utc=True)], 'text': ['This', 'should', 'fail']}
    testdata = DataFrame(data)
    ax = testdata.plot(y=col)
    result = ax.get_lines()[0].get_data()[1]
    expected = testdata[col].values
    assert (result == expected).all()