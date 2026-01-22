import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_grouping_grouper(self, data_for_grouping):
    df = pd.DataFrame({'A': ['B', 'B', None, None, 'A', 'A', 'B'], 'B': data_for_grouping})
    gr1 = df.groupby('A').grouper.groupings[0]
    gr2 = df.groupby('B').grouper.groupings[0]
    tm.assert_numpy_array_equal(gr1.grouping_vector, df.A.values)
    tm.assert_extension_array_equal(gr2.grouping_vector, data_for_grouping)