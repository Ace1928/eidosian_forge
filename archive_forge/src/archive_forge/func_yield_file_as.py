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
def yield_file_as(self: TDF, name: str) -> None:
    """Cache the dataframe in file

        :param name: the name of the yielded dataframe

        .. note::

            In only the following cases you can yield file/table:

            * you have not checkpointed (persisted) the dataframe, for example
              ``df.yield_file_as("a")``
            * you have used :meth:`~.deterministic_checkpoint`, for example
              ``df.deterministic_checkpoint().yield_file_as("a")``
            * yield is workflow, compile level logic

            For the first case, the yield will also be a strong checkpoint so
            whenever you yield a dataframe as a file, the dataframe has been saved as a
            file and loaded back as a new dataframe.
        """
    if not self._task.has_checkpoint:
        self.deterministic_checkpoint(storage_type='file', namespace=str(uuid4()))
    self.workflow._yields[name] = self._task.yielded