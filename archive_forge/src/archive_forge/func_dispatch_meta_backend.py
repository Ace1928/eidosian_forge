import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def dispatch_meta_backend(matrix: DMatrix, data: DataType, name: str, dtype: Optional[NumpyDType]=None) -> None:
    """Dispatch for meta info."""
    handle = matrix.handle
    assert handle is not None
    _validate_meta_shape(data, name)
    if data is None:
        return
    if _is_list(data):
        _meta_from_list(data, name, dtype, handle)
        return
    if _is_tuple(data):
        _meta_from_tuple(data, name, dtype, handle)
        return
    if _is_np_array_like(data):
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_pandas_df(data):
        data, _, _ = _transform_pandas_df(data, False, meta=name, meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_pandas_series(data):
        _meta_from_pandas_series(data, name, dtype, handle)
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cupy_array(data):
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cudf_ser(data):
        _meta_from_cudf_series(data, name, handle)
        return
    if _is_cudf_df(data):
        _meta_from_cudf_df(data, name, handle)
        return
    if _is_dt_df(data):
        _meta_from_dt(data, name, dtype, handle)
        return
    if _is_modin_df(data):
        data, _, _ = _transform_pandas_df(data, False, meta=name, meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_modin_series(data):
        data = data.values.astype('float')
        assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _has_array_protocol(data):
        array = np.asarray(data)
        _meta_from_numpy(array, name, dtype, handle)
        return
    raise TypeError('Unsupported type for ' + name, str(type(data)))