from statsmodels.compat.pandas import PD_LT_2_2_0
from datetime import datetime
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
def test_keyerror_start_date():
    x = np.arange(1, 36.0)
    dates = pd.date_range('1972-4-30', '2006-4-30', freq=YE_APR)
    series = pd.Series(x, index=dates)
    model = TimeSeriesModel(series)
    npt.assert_raises(KeyError, model._get_prediction_index, '1970-4-30', None)