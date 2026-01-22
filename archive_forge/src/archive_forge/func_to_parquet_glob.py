from __future__ import annotations
import inspect
import pathlib
import pickle
from typing import IO, AnyStr, Callable, Iterator, Literal, Optional, Union
import pandas
import pandas._libs.lib as lib
from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, StorageOptions
from modin.core.storage_formats import BaseQueryCompiler
from modin.utils import expanduser_path_arg
from . import DataFrame
@expanduser_path_arg('path')
def to_parquet_glob(self, path, engine='auto', compression='snappy', index=None, partition_cols=None, storage_options: StorageOptions=None, **kwargs) -> None:
    """
    Write a DataFrame to the binary parquet format.

    This experimental feature provides parallel writing into multiple parquet files which are
    defined by glob pattern, otherwise (without glob pattern) default pandas implementation is used.

    Notes
    -----
    * Only string type supported for `path` argument.
    * The rest of the arguments are the same as for `pandas.to_parquet`.
    """
    obj = self
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    if isinstance(self, DataFrame):
        obj = self._query_compiler
    FactoryDispatcher.to_parquet_glob(obj, path=path, engine=engine, compression=compression, index=index, partition_cols=partition_cols, storage_options=storage_options, **kwargs)