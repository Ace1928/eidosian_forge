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
def map_bag(self, bag: Bag, map_func: Callable[[BagPartitionCursor, LocalBag], LocalBag], partition_spec: PartitionSpec, on_init: Optional[Callable[[int, Bag], Any]]=None) -> Bag:
    """Apply a function to each partition after you partition the bag in a
        specified way.

        :param df: input dataframe
        :param map_func: the function to apply on every logical partition
        :param partition_spec: partition specification
        :param on_init: callback function when the physical partition is initializaing,
          defaults to None
        :return: the bag after the map operation
        """
    raise NotImplementedError