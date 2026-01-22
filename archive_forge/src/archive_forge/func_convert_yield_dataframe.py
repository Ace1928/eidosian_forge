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
def convert_yield_dataframe(self, df: DataFrame, as_local: bool) -> DataFrame:
    """Convert a yield dataframe to a dataframe that can be used after this
        execution engine stops.

        :param df: DataFrame
        :param as_local: whether yield a local dataframe
        :return: another DataFrame that can be used after this execution engine stops

        .. note::

            By default, the output dataframe is the input dataframe. But it should be
            overridden if when an engine stops and the input dataframe will become
            invalid.

            For example, if you custom a spark engine where you start and stop the spark
            session in this engine's :meth:`~.start_engine` and :meth:`~.stop_engine`,
            then the spark dataframe will be invalid. So you may consider converting
            it to a local dataframe so it can still exist after the engine stops.
        """
    return df.as_local() if as_local else df