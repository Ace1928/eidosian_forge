import functools
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
def test_read_cell_annotation():
    expected = pd.DataFrame(['test', np.nan, 'test 3'], columns=['Column 1'])
    result = pd.read_excel('test_cell_annotation.ods')
    tm.assert_frame_equal(result, expected)