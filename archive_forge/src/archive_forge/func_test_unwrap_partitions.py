import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
@pytest.mark.parametrize('axis', [None, 0, 1])
@pytest.mark.parametrize('reverse_index', [True, False])
@pytest.mark.parametrize('reverse_columns', [True, False])
def test_unwrap_partitions(axis, reverse_index, reverse_columns):
    data = test_data['int_data']

    def get_df(lib, data):
        df = lib.DataFrame(data)
        if reverse_index:
            df.index = df.index[::-1]
        if reverse_columns:
            df.columns = df.columns[::-1]
        return df
    df = get_df(pd, data)
    expected_df = pd.DataFrame(get_df(pandas, data))
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    if axis is None:
        actual_partitions = np.array(unwrap_partitions(df, axis=axis))
        assert expected_partitions.shape == actual_partitions.shape
        for row_idx in range(expected_partitions.shape[0]):
            for col_idx in range(expected_partitions.shape[1]):
                df_equals(get_func(expected_partitions[row_idx][col_idx].list_of_blocks[0]), get_func(actual_partitions[row_idx][col_idx]))
    else:
        expected_axis_partitions = expected_df._query_compiler._modin_frame._partition_mgr_cls.axis_partition(expected_partitions, axis ^ 1)
        expected_axis_partitions = [axis_partition.force_materialization().unwrap(squeeze=True) for axis_partition in expected_axis_partitions]
        actual_axis_partitions = unwrap_partitions(df, axis=axis)
        assert len(expected_axis_partitions) == len(actual_axis_partitions)
        for item_idx in range(len(expected_axis_partitions)):
            if Engine.get() in ['Ray', 'Dask', 'Unidist']:
                df_equals(get_func(expected_axis_partitions[item_idx]), get_func(actual_axis_partitions[item_idx]))