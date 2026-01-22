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
def assert_not_eq(self, *dfs: Any, **params: Any) -> None:
    """Assert if all dataframes are not equal to the first dataframe.
        It's for internal, unit test purpose only. It will convert both dataframes to
        :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame`, so it assumes
        all dataframes are small and fast enough to convert. DO NOT use it
        on critical or expensive tasks.

        :param dfs: |DataFramesLikeObject|
        :param digits: precision on float number comparison, defaults to 8
        :param check_order: if to compare the row orders, defaults to False
        :param check_schema: if compare schemas, defaults to True
        :param check_content: if to compare the row values, defaults to True
        :param no_pandas: if true, it will compare the string representations of the
          dataframes, otherwise, it will convert both to pandas dataframe to compare,
          defaults to False

        :raises AssertionError: if any dataframe equals to the first dataframe
        """
    self.output(*dfs, using=AssertNotEqual, params=params)