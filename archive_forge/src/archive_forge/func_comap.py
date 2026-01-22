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
def comap(self, df: DataFrame, map_func: Callable[[PartitionCursor, DataFrames], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrames], Any]]=None):
    """Apply a function to each zipped partition on the zipped dataframe.

        :param df: input dataframe, it must be a zipped dataframe (it has to be a
          dataframe output from :meth:`~.zip` or :meth:`~.zip_all`)
        :param map_func: the function to apply on every zipped partition
        :param output_schema: |SchemaLikeObject| that can't be None.
          Please also understand :ref:`why we need this
          <tutorial:tutorials/beginner/interface:schema>`
        :param partition_spec: partition specification for processing the zipped
          zipped dataframe.
        :param on_init: callback function when the physical partition is initializaing,
          defaults to None
        :return: the dataframe after the comap operation

        .. note::

            * The input of this method must be an output of :meth:`~.zip` or
              :meth:`~.zip_all`
            * The ``partition_spec`` here is NOT related with how you zipped the
              dataframe and however you set it, will only affect the processing speed,
              actually the partition keys will be overriden to the zipped dataframe
              partition keys. You may use it in this way to improve the efficiency:
              ``PartitionSpec(algo="even", num="ROWCOUNT")``,
              this tells the execution engine to put each zipped partition into a
              physical partition so it can achieve the best possible load balance.
            * If input dataframe has keys, the dataframes you get in ``map_func`` and
              ``on_init`` will have keys, otherwise you will get list-like dataframes
            * on_init function will get a DataFrames object that has the same structure,
              but has all empty dataframes, you can use the schemas but not the data.

        .. seealso::

            For more details and examples, read |ZipComap|
        """
    assert_or_throw(df.metadata['serialized'], ValueError('df is not serilaized'))
    key_schema = df.schema - _FUGUE_SERIALIZED_BLOB_SCHEMA
    cs = _Comap(df, key_schema, map_func, output_schema, on_init)
    partition_spec = PartitionSpec(partition_spec, by=key_schema.names + [_FUGUE_SERIALIZED_BLOB_DUMMY_COL], presort=_FUGUE_SERIALIZED_BLOB_NO_COL)
    return self.map_engine.map_dataframe(df, cs.run, output_schema, partition_spec, on_init=cs.on_init)