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
def test_pivot_complex_aggfunc(self, data):
    f = {'D': ['std'], 'E': ['sum']}
    expected = data.groupby(['A', 'B']).agg(f).unstack('B')
    result = data.pivot_table(index='A', columns='B', aggfunc=f)
    tm.assert_frame_equal(result, expected)