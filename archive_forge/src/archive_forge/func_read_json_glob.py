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
@expanduser_path_arg('path_or_buf')
def read_json_glob(path_or_buf, *, orient: str | None=None, typ: Literal['frame', 'series']='frame', dtype: DtypeArg | None=None, convert_axes=None, convert_dates: bool | list[str]=True, keep_default_dates: bool=True, precise_float: bool=False, date_unit: str | None=None, encoding: str | None=None, encoding_errors: str | None='strict', lines: bool=False, chunksize: int | None=None, compression: CompressionOptions='infer', nrows: int | None=None, storage_options: StorageOptions=None, dtype_backend: Union[DtypeBackend, lib.NoDefault]=lib.no_default, engine='ujson') -> DataFrame:
    """
    Convert a JSON string to pandas object.

    This experimental feature provides parallel reading from multiple json files which are
    defined by glob pattern. The files must contain parts of one dataframe, which can be
    obtained, for example, by `DataFrame.modin.to_json_glob` function.

    Returns
    -------
    DataFrame

    Notes
    -----
    * Only string type supported for `path_or_buf` argument.
    * The rest of the arguments are the same as for `pandas.read_json`.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    if nrows is not None:
        raise NotImplementedError('`read_json_glob` only support nrows is None, otherwise use `to_json`.')
    return DataFrame(query_compiler=FactoryDispatcher.read_json_glob(path_or_buf=path_or_buf, orient=orient, typ=typ, dtype=dtype, convert_axes=convert_axes, convert_dates=convert_dates, keep_default_dates=keep_default_dates, precise_float=precise_float, date_unit=date_unit, encoding=encoding, encoding_errors=encoding_errors, lines=lines, chunksize=chunksize, compression=compression, nrows=nrows, storage_options=storage_options, dtype_backend=dtype_backend, engine=engine))