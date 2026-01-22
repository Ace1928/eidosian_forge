import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_update_df(self):
    """Ensure that `update_df` updates the df that the pipeline runs on."""
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df + 3, is_output=True)
    new_df = df * -1
    pipeline.update_df(new_df)
    output_df = pipeline.compute_batch()[0]
    df_equals(df * -1 + 3, output_df)