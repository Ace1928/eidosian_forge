import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_fillna_fill_other(self, data):
    result = pd.DataFrame({'A': data, 'B': [np.nan] * len(data)}).fillna({'B': 0.0})
    expected = pd.DataFrame({'A': data, 'B': [0.0] * len(result)})
    tm.assert_frame_equal(result, expected)