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
def test_pivot_table_nocols(self):
    df = DataFrame({'rows': ['a', 'b', 'c'], 'cols': ['x', 'y', 'z'], 'values': [1, 2, 3]})
    rs = df.pivot_table(columns='cols', aggfunc='sum')
    xp = df.pivot_table(index='cols', aggfunc='sum').T
    tm.assert_frame_equal(rs, xp)
    rs = df.pivot_table(columns='cols', aggfunc={'values': 'mean'})
    xp = df.pivot_table(index='cols', aggfunc={'values': 'mean'}).T
    tm.assert_frame_equal(rs, xp)