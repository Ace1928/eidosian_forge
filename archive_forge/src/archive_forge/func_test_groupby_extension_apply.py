import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    df.groupby('B', group_keys=False).apply(groupby_apply_op)
    df.groupby('B', group_keys=False).A.apply(groupby_apply_op)
    df.groupby('A', group_keys=False).apply(groupby_apply_op)
    df.groupby('A', group_keys=False).B.apply(groupby_apply_op)