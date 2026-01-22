from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_na_trailing_columns(all_parsers):
    parser = all_parsers
    data = 'Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax\n2012-03-14,USD,AAPL,BUY,1000\n2012-05-12,USD,SBUX,SELL,500'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([['2012-03-14', 'USD', 'AAPL', 'BUY', 1000, np.nan, np.nan, np.nan], ['2012-05-12', 'USD', 'SBUX', 'SELL', 500, np.nan, np.nan, np.nan]], columns=['Date', 'Currency', 'Symbol', 'Type', 'Units', 'UnitPrice', 'Cost', 'Tax'])
    tm.assert_frame_equal(result, expected)