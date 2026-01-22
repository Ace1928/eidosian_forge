import sys
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import PYPY
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_memory_usage_components_series(series_with_simple_index):
    series = series_with_simple_index
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage