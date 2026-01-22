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
def trades(self):
    df = pd.DataFrame([['20160525 13:30:00.023', 'MSFT', '51.9500', '75', 'NASDAQ'], ['20160525 13:30:00.038', 'MSFT', '51.9500', '155', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.7700', '100', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9200', '100', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '200', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '300', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '600', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '44', 'NASDAQ'], ['20160525 13:30:00.074', 'AAPL', '98.6700', '478343', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6700', '478343', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6600', '6', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '30', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '75', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '20', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '35', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '10', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.5500', '6', 'ARCA'], ['20160525 13:30:00.075', 'AAPL', '98.5500', '6', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '1000', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '200', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '300', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '400', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '600', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '200', 'ARCA'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '783', 'NASDAQ'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '100', 'NASDAQ'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '100', 'NASDAQ']], columns='time,ticker,price,quantity,marketCenter'.split(','))
    df['price'] = df['price'].astype('float64')
    df['quantity'] = df['quantity'].astype('int64')
    return self.prep_data(df)