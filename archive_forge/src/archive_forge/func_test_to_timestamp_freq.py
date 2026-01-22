from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_freq(self):
    idx = period_range('2017', periods=12, freq='Y-DEC')
    result = idx.to_timestamp()
    expected = date_range('2017', periods=12, freq='YS-JAN')
    tm.assert_index_equal(result, expected)