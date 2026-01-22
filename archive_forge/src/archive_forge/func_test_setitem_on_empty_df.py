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
@pytest.mark.skipif(StorageFormat.get() == 'Hdk', reason='https://github.com/intel-ai/hdk/issues/165')
@pytest.mark.parametrize('data', [{}, {'id': [], 'max_speed': [], 'health': []}, {'id': [1], 'max_speed': [2], 'health': [3]}, {'id': [4, 40, 400], 'max_speed': [111, 222, 333], 'health': [33, 22, 11]}], ids=['empty_frame', 'empty_cols', '1_length_cols', '2_length_cols'])
@pytest.mark.parametrize('value', [[11, 22], [11, 22, 33]], ids=['2_length_val', '3_length_val'])
@pytest.mark.parametrize('convert_to_series', [False, True])
@pytest.mark.parametrize('new_col_id', [123, 'new_col'], ids=['integer', 'string'])
def test_setitem_on_empty_df(data, value, convert_to_series, new_col_id):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    def applyier(df):
        if convert_to_series:
            converted_value = pandas.Series(value) if isinstance(df, pandas.DataFrame) else pd.Series(value)
        else:
            converted_value = value
        df[new_col_id] = converted_value
        return df
    expected_exception = None
    if not convert_to_series:
        values_length = len(value)
        index_length = len(pandas_df.index)
        expected_exception = ValueError(f'Length of values ({values_length}) does not match length of index ({index_length})')
    eval_general(modin_df, pandas_df, applyier, comparator_kwargs={'check_dtypes': not (len(pandas_df) == 0 and len(pandas_df.columns) != 0)}, expected_exception=expected_exception)