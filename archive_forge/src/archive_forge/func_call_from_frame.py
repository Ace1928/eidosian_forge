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
def call_from_frame(df):
    if type(df).__module__.startswith('pandas'):
        return pandas.MultiIndex.from_frame(df, sortorder)
    else:
        return pd.MultiIndex.from_frame(df, sortorder)