import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
@pytest.mark.parametrize('columns', ['original_col', 'new_col'])
@pytest.mark.parametrize('index', ['original_idx', 'new_idx'])
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_from_partitions_mismatched_labels(axis, index, columns):
    expected_df = pd.DataFrame(test_data['int_data'])
    partitions = unwrap_partitions(expected_df, axis=axis)
    index = expected_df.index if index == 'original_idx' else [f'row{i}' for i in expected_df.index]
    columns = expected_df.columns if columns == 'original_col' else [f'col{i}' for i in expected_df.columns]
    expected_df.index = index
    expected_df.columns = columns
    actual_df = from_partitions(partitions, axis=axis, index=index, columns=columns)
    df_equals(expected_df, actual_df)