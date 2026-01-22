import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_stable_sort():
    df = DataFrame(data=[(Timestamp('2024-01-30 13:00:00'), 13.0), (Timestamp('2024-01-30 13:00:00'), 13.1), (Timestamp('2024-01-30 12:00:00'), 12.0), (Timestamp('2024-01-30 12:00:00'), 12.1)], columns=['dt', 'value']).set_index(['dt'])
    result = df.sort_index(level='dt', kind='stable')
    expected = DataFrame(data=[(Timestamp('2024-01-30 12:00:00'), 12.0), (Timestamp('2024-01-30 12:00:00'), 12.1), (Timestamp('2024-01-30 13:00:00'), 13.0), (Timestamp('2024-01-30 13:00:00'), 13.1)], columns=['dt', 'value']).set_index(['dt'])
    tm.assert_frame_equal(result, expected)