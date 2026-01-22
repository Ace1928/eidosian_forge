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
def test___setitem__single_item_in_series():
    modin_series = pd.Series(99)
    pandas_series = pandas.Series(99)
    modin_series[:1] = pd.Series(100)
    pandas_series[:1] = pandas.Series(100)
    df_equals(modin_series, pandas_series)