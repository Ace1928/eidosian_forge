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
def test_rename_issue5600():
    df = pd.DataFrame({'a': [1, 2]})
    df_renamed = df.rename(columns={'a': 'new_a'}, copy=True, inplace=False)
    assert df.dtypes.keys().tolist() == ['a']
    assert df.columns.tolist() == ['a']
    assert df_renamed.dtypes.keys().tolist() == ['new_a']
    assert df_renamed.columns.tolist() == ['new_a']