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
def test_pivot_multi_functions(self, data):
    f = lambda func: pivot_table(data, values=['D', 'E'], index=['A', 'B'], columns='C', aggfunc=func)
    result = f(['mean', 'std'])
    means = f('mean')
    stds = f('std')
    expected = concat([means, stds], keys=['mean', 'std'], axis=1)
    tm.assert_frame_equal(result, expected)
    f = lambda func: pivot_table(data, values=['D', 'E'], index=['A', 'B'], columns='C', aggfunc=func, margins=True)
    result = f(['mean', 'std'])
    means = f('mean')
    stds = f('std')
    expected = concat([means, stds], keys=['mean', 'std'], axis=1)
    tm.assert_frame_equal(result, expected)