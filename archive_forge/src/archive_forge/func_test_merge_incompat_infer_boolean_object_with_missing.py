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
def test_merge_incompat_infer_boolean_object_with_missing(self):
    df1 = DataFrame({'key': Series([True, False, np.nan], dtype=object)})
    df2 = DataFrame({'key': [True, False]})
    expected = DataFrame({'key': [True, False]}, dtype=object)
    result = merge(df1, df2, on='key')
    tm.assert_frame_equal(result, expected)
    result = merge(df2, df1, on='key')
    tm.assert_frame_equal(result, expected)