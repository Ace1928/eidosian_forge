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
def test_iloc_loc_key_length_except():
    modin_ser, pandas_ser = (pd.Series(0), pandas.Series(0))
    eval_general(modin_ser, pandas_ser, lambda ser: ser.iloc[0, 0], expected_exception=pandas.errors.IndexingError('Too many indexers'))
    eval_general(modin_ser, pandas_ser, lambda ser: ser.loc[0, 0], expected_exception=pandas.errors.IndexingError('Too many indexers'))