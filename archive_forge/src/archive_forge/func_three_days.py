import numpy as np
import pytest
import pandas as pd
from pandas import Index
@pytest.fixture(params=[pd.offsets.Day(3), pd.offsets.Hour(72), pd.Timedelta(days=3).to_pytimedelta(), pd.Timedelta('72:00:00'), np.timedelta64(3, 'D'), np.timedelta64(72, 'h')], ids=lambda x: type(x).__name__)
def three_days(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 3-day timedelta
    """
    return request.param