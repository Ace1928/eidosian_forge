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
def test_rename_inplace():
    source_df = pandas.DataFrame(test_data['int_data'])[['col1', 'index', 'col3', 'col4']]
    modin_df = pd.DataFrame(source_df)
    df_equals(modin_df.rename(columns={'col3': 'foo'}), source_df.rename(columns={'col3': 'foo'}))
    frame = source_df.copy()
    modin_frame = modin_df.copy()
    frame.rename(columns={'col3': 'foo'}, inplace=True)
    modin_frame.rename(columns={'col3': 'foo'}, inplace=True)
    df_equals(modin_frame, frame)