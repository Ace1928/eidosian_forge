import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_slice(self, df):
    msg = 'cannot do slice indexing on CategoricalIndex with these indexers \\[1\\] of type int'
    with pytest.raises(TypeError, match=msg):
        df.loc[1:5]
    result = df.loc['b':'c']
    expected = df.iloc[[2, 3, 4]]
    tm.assert_frame_equal(result, expected)