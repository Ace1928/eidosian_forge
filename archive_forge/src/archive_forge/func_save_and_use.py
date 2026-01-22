import sys
from collections import defaultdict
from typing import (
from uuid import uuid4
from adagio.specs import WorkflowSpec
from triad import (
from fugue._utils.exception import modify_traceback
from fugue.collections.partition import PartitionSpec
from fugue.collections.sql import StructuredRawSQL
from fugue.collections.yielded import Yielded
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.column import all_cols, col, lit
from fugue.constants import (
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, YieldedDataFrame
from fugue.dataframe.api import is_df
from fugue.dataframe.dataframes import DataFrames
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowError
from fugue.execution.api import engine_context
from fugue.extensions._builtins import (
from fugue.extensions.transformer.convert import _to_output_transformer, _to_transformer
from fugue.rpc import to_rpc_handler
from fugue.rpc.base import EmptyRPCHandler
from fugue.workflow._checkpoint import StrongCheckpoint, WeakCheckpoint
from fugue.workflow._tasks import Create, FugueTask, Output, Process
from fugue.workflow._workflow_context import FugueWorkflowContext
def save_and_use(self: TDF, path: str, fmt: str='', mode: str='overwrite', partition: Any=None, single: bool=False, **kwargs: Any) -> TDF:
    """Save this dataframe to a persistent storage and load back to use
        in the following steps

        :param path: output path
        :param fmt: format hint can accept ``parquet``, ``csv``, ``json``,
          defaults to None, meaning to infer
        :param mode: can accept ``overwrite``, ``append``, ``error``,
          defaults to "overwrite"
        :param partition: |PartitionLikeObject|, how to partition the
          dataframe before saving, defaults to empty
        :param single: force the output as a single file, defaults to False
        :param kwargs: parameters to pass to the underlying framework

        For more details and examples, read
        :ref:`Save & Load <tutorial:tutorials/advanced/dag:save & load>`.
        """
    if partition is None:
        partition = self.partition_spec
    df = self.workflow.process(self, using=SaveAndUse, pre_partition=partition, params=dict(path=path, fmt=fmt, mode=mode, single=single, params=kwargs))
    return self._to_self_type(df)