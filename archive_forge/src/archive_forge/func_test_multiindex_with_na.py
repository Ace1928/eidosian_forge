import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_with_na(self):
    df = DataFrame([['A', np.nan, 1.23, 4.56], ['A', 'G', 1.23, 4.56], ['A', 'D', 9.87, 10.54]], columns=['pivot_0', 'pivot_1', 'col_1', 'col_2']).set_index(['pivot_0', 'pivot_1'])
    df.at[('A', 'F'), 'col_2'] = 0.0
    expected = DataFrame([['A', np.nan, 1.23, 4.56], ['A', 'G', 1.23, 4.56], ['A', 'D', 9.87, 10.54], ['A', 'F', np.nan, 0.0]], columns=['pivot_0', 'pivot_1', 'col_1', 'col_2']).set_index(['pivot_0', 'pivot_1'])
    tm.assert_frame_equal(df, expected)