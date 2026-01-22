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
def weak_checkpoint(self: TDF, lazy: bool=False, **kwargs: Any) -> TDF:
    """Cache the dataframe in memory

        :param lazy: whether it is a lazy checkpoint, defaults to False (eager)
        :param kwargs: paramteters for the underlying execution engine function
        :return: the cached dataframe

        .. note::

            Weak checkpoint in most cases is the best choice for caching a dataframe to
            avoid duplicated computation. However it does not guarantee to break up the
            the compute dependency for this dataframe, so when you have very complicated
            compute, you may encounter issues such as stack overflow. Also, weak
            checkpoint normally caches the dataframe in memory, if memory is a concern,
            then you should consider :meth:`~.strong_checkpoint`
        """
    self._task.set_checkpoint(WeakCheckpoint(lazy=lazy, **kwargs))
    return self