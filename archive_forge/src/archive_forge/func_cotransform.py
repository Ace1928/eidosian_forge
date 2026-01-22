from typing import List, no_type_check
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw
from triad.utils.convert import to_type
from fugue.collections.partition import PartitionCursor
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.utils import _df_eq
from fugue.exceptions import FugueWorkflowError
from fugue.execution.execution_engine import (
from fugue.rpc import EmptyRPCHandler, to_rpc_handler
from ..outputter import Outputter
from ..transformer.convert import _to_output_transformer
from ..transformer.transformer import CoTransformer, Transformer
@no_type_check
def cotransform(self, df: DataFrame, tf: CoTransformer) -> None:
    assert_or_throw(df.metadata.get('serialized', False), 'must use serialized dataframe')
    tf._key_schema = df.schema - _FUGUE_SERIALIZED_BLOB_SCHEMA
    empty_dfs = _generate_comap_empty_dfs(df.metadata['schemas'], df.metadata.get('serialized_has_name', False))
    tf._output_schema = Schema(tf.get_output_schema(empty_dfs))
    tr = _CoTransformerRunner(df, tf, self._ignore_errors)
    df = self.execution_engine.comap(df=df, map_func=tr.run, output_schema=tf.output_schema, partition_spec=tf.partition_spec, on_init=tr.on_init)
    self.execution_engine.persist(df, lazy=False)