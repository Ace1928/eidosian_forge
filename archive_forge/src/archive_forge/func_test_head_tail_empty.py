import numpy as np
from pandas import DataFrame
import pandas._testing as tm
def test_head_tail_empty():
    empty_df = DataFrame()
    tm.assert_frame_equal(empty_df.tail(), empty_df)
    tm.assert_frame_equal(empty_df.head(), empty_df)