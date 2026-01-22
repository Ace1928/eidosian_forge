import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import uuid4
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw, to_uuid
from triad.collections.function_wrapper import AnnotatedParam
from triad.exceptions import InvalidOperationError
from triad.utils.convert import to_size
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.collections.yielded import PhysicalYielded, Yielded
from fugue.column import (
from fugue.constants import _FUGUE_GLOBAL_CONF, FUGUE_SQL_DEFAULT_DIALECT
from fugue.dataframe import AnyDataFrame, DataFrame, DataFrames, fugue_annotated_param
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import LocalDataFrame
from fugue.dataframe.utils import deserialize_df, serialize_df
from fugue.exceptions import FugueWorkflowRuntimeError
def save_table(self, df: DataFrame, table: str, mode: str='overwrite', partition_spec: Optional[PartitionSpec]=None, **kwargs: Any) -> None:
    """Save the dataframe to a table

        :param df: the dataframe to save
        :param table: the table name
        :param mode: can accept ``overwrite``, ``error``,
          defaults to "overwrite"
        :param partition_spec: how to partition the dataframe before saving,
          defaults None
        :param kwargs: parameters to pass to the underlying framework
        """
    raise NotImplementedError