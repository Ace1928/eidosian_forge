import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def dispatch_proxy_set_data(proxy: _ProxyDMatrix, data: DataType, cat_codes: Optional[list], allow_host: bool) -> None:
    """Dispatch for QuantileDMatrix."""
    if not _is_cudf_ser(data) and (not _is_pandas_series(data)):
        _check_data_shape(data)
    if _is_cudf_df(data):
        proxy._set_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cudf_ser(data):
        proxy._set_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cupy_array(data):
        proxy._set_data_from_cuda_interface(data)
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._set_data_from_cuda_interface(data)
        return
    err = TypeError('Value type is not supported for data iterator:' + str(type(data)))
    if not allow_host:
        raise err
    if _is_np_array_like(data):
        _check_data_shape(data)
        proxy._set_data_from_array(data)
        return
    if _is_scipy_csr(data):
        proxy._set_data_from_csr(data)
        return
    raise err