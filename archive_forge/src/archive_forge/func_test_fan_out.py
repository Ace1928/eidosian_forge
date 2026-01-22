import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_fan_out(self):
    """Check that the fan_out argument is appropriately handled."""
    df = pd.DataFrame([[0, 1, 2]])

    def new_col_adder(df, partition_id):
        df['new_col'] = partition_id
        return df

    def reducer(dfs):
        new_cols = ''.join([str(df['new_col'].values[0]) for df in dfs])
        dfs[0]['new_col1'] = new_cols
        return dfs[0]
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(new_col_adder, fan_out=True, reduce_fn=reducer, pass_partition_id=True, is_output=True)
    new_df = pipeline.compute_batch()[0]
    correct_df = pd.DataFrame([[0, 1, 2]])
    correct_df['new_col'] = 0
    correct_df['new_col1'] = ''.join([str(i) for i in range(NPartitions.get())])
    df_equals(correct_df, new_df)
    partition1 = RayWrapper.put(pandas.DataFrame([[0, 1, 2]]))
    partition2 = RayWrapper.put(pandas.DataFrame([[3, 4, 5]]))
    df = from_partitions([partition1, partition2], 0)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(new_col_adder, fan_out=True, reduce_fn=reducer, pass_partition_id=True, is_output=True)
    with pytest.raises(NotImplementedError, match='Fan out is only supported with DataFrames with 1 partition.'):
        pipeline.compute_batch()[0]