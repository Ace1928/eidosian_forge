import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.fixture
def quotes(self):
    df = pd.DataFrame([['20160525 13:30:00.023', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.023', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.041', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.072', 'GOOG', '720.50', '720.88'], ['20160525 13:30:00.075', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.92', '51.95']], columns='time,ticker,bid,ask'.split(','))
    df['bid'] = df['bid'].astype('float64')
    df['ask'] = df['ask'].astype('float64')
    return self.prep_data(df, dedupe=True)