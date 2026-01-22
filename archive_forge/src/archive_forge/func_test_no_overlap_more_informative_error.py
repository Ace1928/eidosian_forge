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
def test_no_overlap_more_informative_error(self):
    dt = datetime.now()
    df1 = DataFrame({'x': ['a']}, index=[dt])
    df2 = DataFrame({'y': ['b', 'c']}, index=[dt, dt])
    msg = f'No common columns to perform merge on. Merge options: left_on={None}, right_on={None}, left_index={False}, right_index={False}'
    with pytest.raises(MergeError, match=msg):
        merge(df1, df2)