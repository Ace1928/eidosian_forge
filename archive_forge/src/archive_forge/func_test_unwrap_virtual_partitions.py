import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
def test_unwrap_virtual_partitions():
    data = test_data['int_data']
    df = pd.DataFrame(data)
    virtual_partitioned_df = pd.concat([df] * 10)
    actual_partitions = np.array(unwrap_partitions(virtual_partitioned_df, axis=None))
    expected_df = pd.concat([pd.DataFrame(data)] * 10)
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    assert expected_partitions.shape == actual_partitions.shape
    for row_idx in range(expected_partitions.shape[0]):
        for col_idx in range(expected_partitions.shape[1]):
            df_equals(get_func(expected_partitions[row_idx][col_idx].force_materialization().list_of_blocks[0]), get_func(actual_partitions[row_idx][col_idx]))