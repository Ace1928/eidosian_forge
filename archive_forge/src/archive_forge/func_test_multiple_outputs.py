import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_multiple_outputs(self):
    """Create a pipeline with multiple outputs, and check that all are computed correctly."""
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True)
    pipeline.add_query(lambda df: df + 30, is_output=True)
    new_dfs = pipeline.compute_batch()
    assert len(new_dfs) == 3, 'Pipeline did not return all outputs'
    correct_df = pd.DataFrame(arr) * -30
    df_equals(correct_df, new_dfs[0])
    correct_df = correct_df.rename(columns={i: f'col {i}' for i in range(1000)})
    df_equals(correct_df, new_dfs[1])
    correct_df += 30
    df_equals(correct_df, new_dfs[2])