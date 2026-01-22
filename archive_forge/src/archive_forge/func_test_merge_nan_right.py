from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_nan_right(self):
    df1 = DataFrame({'i1': [0, 1], 'i2': [0, 1]})
    df2 = DataFrame({'i1': [0], 'i3': [0]})
    result = df1.join(df2, on='i1', rsuffix='_')
    expected = DataFrame({'i1': {0: 0.0, 1: 1}, 'i2': {0: 0, 1: 1}, 'i1_': {0: 0, 1: np.nan}, 'i3': {0: 0.0, 1: np.nan}, None: {0: 0, 1: 0}}, columns=Index(['i1', 'i2', 'i1_', 'i3', None], dtype=object)).set_index(None).reset_index()[['i1', 'i2', 'i1_', 'i3']]
    result.columns = result.columns.astype('object')
    tm.assert_frame_equal(result, expected, check_dtype=False)