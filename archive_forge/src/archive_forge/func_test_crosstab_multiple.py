import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_multiple(self, df):
    result = crosstab(df['A'], [df['B'], df['C']])
    expected = df.groupby(['A', 'B', 'C']).size()
    expected = expected.unstack('B').unstack('C').fillna(0).astype(np.int64)
    tm.assert_frame_equal(result, expected)
    result = crosstab([df['B'], df['C']], df['A'])
    expected = df.groupby(['B', 'C', 'A']).size()
    expected = expected.unstack('A').fillna(0).astype(np.int64)
    tm.assert_frame_equal(result, expected)