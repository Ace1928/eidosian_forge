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
def test_pivot_multi_values(self, data):
    result = pivot_table(data, values=['D', 'E'], index='A', columns=['B', 'C'], fill_value=0)
    expected = pivot_table(data.drop(['F'], axis=1), index='A', columns=['B', 'C'], fill_value=0)
    tm.assert_frame_equal(result, expected)