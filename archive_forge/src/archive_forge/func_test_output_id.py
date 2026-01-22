import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_output_id(self):
    """Ensure `output_id` is handled correctly when passed."""
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df, 0)
    pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
    with pytest.raises(ValueError, match='Output ID must be specified for all nodes.'):
        pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True)
    assert len(pipeline.query_list) == 0 and len(pipeline.outputs) == 1, 'Invalid `add_query` incorrectly added a node to the pipeline.'
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True)
    with pytest.raises(ValueError, match='Output ID must be specified for all nodes.'):
        pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True, output_id=20)
    assert len(pipeline.query_list) == 0 and len(pipeline.outputs) == 1, 'Invalid `add_query` incorrectly added a node to the pipeline.'
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df, is_output=True)
    with pytest.raises(ValueError, match='`pass_output_id` is set to True, but output ids have not been specified. ' + 'To pass output ids, please specify them using the `output_id` kwarg with pipeline.add_query'):
        pipeline.compute_batch(postprocessor=lambda df: df, pass_output_id=True)
    with pytest.raises(ValueError, match='Output ID cannot be specified for non-output node.'):
        pipeline.add_query(lambda df: df, output_id=22)
    assert len(pipeline.query_list) == 0 and len(pipeline.outputs) == 1, 'Invalid `add_query` incorrectly added a node to the pipeline.'