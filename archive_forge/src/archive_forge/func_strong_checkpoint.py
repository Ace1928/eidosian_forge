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
def strong_checkpoint(self: TDF, storage_type: str='file', lazy: bool=False, partition: Any=None, single: bool=False, **kwargs: Any) -> TDF:
    """Cache the dataframe as a temporary file

        :param storage_type: can be either ``file`` or ``table``, defaults to ``file``
        :param lazy: whether it is a lazy checkpoint, defaults to False (eager)
        :param partition: |PartitionLikeObject|, defaults to None.
        :param single: force the output as a single file, defaults to False
        :param kwargs: paramteters for the underlying execution engine function
        :return: the cached dataframe

        .. note::

            Strong checkpoint guarantees the output dataframe compute dependency is
            from the temporary file. Use strong checkpoint only when
            :meth:`~.weak_checkpoint` can't be used.

            Strong checkpoint file will be removed after the execution of the workflow.
        """
    self._task.set_checkpoint(StrongCheckpoint(storage_type=storage_type, obj_id=str(uuid4()), deterministic=False, permanent=False, lazy=lazy, partition=partition, single=single, **kwargs))
    return self