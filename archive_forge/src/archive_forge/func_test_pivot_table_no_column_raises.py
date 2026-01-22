from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_table_no_column_raises(self):

    def agg(arr):
        return np.mean(arr)
    df = DataFrame({'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1], 'Z': [10, 20, 30, 40]})
    with pytest.raises(KeyError, match='notpresent'):
        df.pivot_table('notpresent', 'X', 'Y', aggfunc=agg)