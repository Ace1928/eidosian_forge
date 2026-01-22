import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
@pytest.mark.parametrize('data', [[pd.Period('2000', 'D'), pd.Period('2001', 'Y')], [pd.Interval(0, 1, closed='left'), pd.Interval(1, 2, closed='right')], [pd.Timestamp('2000', tz='CET'), pd.Timestamp('2000', tz='UTC')], [pd.Timestamp('2000', tz='CET'), pd.Timestamp('2000')], np.array([pd.Timestamp('2000'), pd.Timestamp('2000', tz='CET')])])
def test_array_inference_fails(data):
    result = pd.array(data)
    expected = NumpyExtensionArray(np.array(data, dtype=object))
    tm.assert_extension_array_equal(result, expected)