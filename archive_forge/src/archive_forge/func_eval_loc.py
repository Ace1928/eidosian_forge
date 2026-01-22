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
def eval_loc(md_df, pd_df, value, key):
    if isinstance(value, tuple):
        assert len(value) == 2
        md_value, pd_value = value
    else:
        md_value, pd_value = (value, value)
    eval_general(md_df, pd_df, lambda df: df.loc.__setitem__(key, pd_value if isinstance(df, pandas.DataFrame) else md_value), __inplace__=True)