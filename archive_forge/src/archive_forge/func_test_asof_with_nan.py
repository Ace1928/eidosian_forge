import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('dates', [['2018-02-27 09:03:30', '2018-02-27 09:04:30'], ['2018-02-27 09:03:00', '2018-02-27 09:05:00']])
@pytest.mark.parametrize('subset', ['a', 'b', ['a', 'b'], None])
def test_asof_with_nan(dates, subset):
    data = {'a': [10, 20, 30, 40, 50], 'b': [None, None, None, None, 500]}
    index = pd.DatetimeIndex(['2018-02-27 09:01:00', '2018-02-27 09:02:00', '2018-02-27 09:03:00', '2018-02-27 09:04:00', '2018-02-27 09:05:00'])
    modin_where = pd.DatetimeIndex(dates)
    pandas_where = pandas.DatetimeIndex(dates)
    compare_asof(data, index, modin_where, pandas_where, subset)