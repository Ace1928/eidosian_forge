import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def dispatch_data_backend(data: DataType, missing: FloatCompatible, threads: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool=False, data_split_mode: DataSplitMode=DataSplitMode.ROW) -> DispatchedDataBackendReturnType:
    """Dispatch data for DMatrix."""
    if not _is_cudf_ser(data) and (not _is_pandas_series(data)):
        _check_data_shape(data)
    if _is_scipy_csr(data):
        return _from_scipy_csr(data, missing, threads, feature_names, feature_types)
    if _is_scipy_csc(data):
        return _from_scipy_csc(data, missing, threads, feature_names, feature_types)
    if _is_scipy_coo(data):
        return _from_scipy_csr(data.tocsr(), missing, threads, feature_names, feature_types)
    if _is_np_array_like(data):
        return _from_numpy_array(data, missing, threads, feature_names, feature_types, data_split_mode)
    if _is_uri(data):
        return _from_uri(data, missing, feature_names, feature_types, data_split_mode)
    if _is_list(data):
        return _from_list(data, missing, threads, feature_names, feature_types)
    if _is_tuple(data):
        return _from_tuple(data, missing, threads, feature_names, feature_types)
    if _is_pandas_series(data):
        import pandas as pd
        data = pd.DataFrame(data)
    if _is_pandas_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads, feature_names, feature_types)
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _from_cudf_df(data, missing, threads, feature_names, feature_types, enable_categorical)
    if _is_cupy_array(data):
        return _from_cupy_array(data, missing, threads, feature_names, feature_types)
    if _is_cupy_csr(data):
        raise TypeError('cupyx CSR is not supported yet.')
    if _is_cupy_csc(data):
        raise TypeError('cupyx CSC is not supported yet.')
    if _is_dlpack(data):
        return _from_dlpack(data, missing, threads, feature_names, feature_types)
    if _is_dt_df(data):
        _warn_unused_missing(data, missing)
        return _from_dt_df(data, missing, threads, feature_names, feature_types, enable_categorical)
    if _is_modin_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads, feature_names, feature_types)
    if _is_modin_series(data):
        return _from_pandas_series(data, missing, threads, enable_categorical, feature_names, feature_types)
    if _is_arrow(data):
        return _from_arrow(data, missing, threads, feature_names, feature_types, enable_categorical)
    if _has_array_protocol(data):
        array = np.asarray(data)
        return _from_numpy_array(array, missing, threads, feature_names, feature_types)
    converted = _convert_unknown_data(data)
    if converted is not None:
        return _from_scipy_csr(converted, missing, threads, feature_names, feature_types)
    raise TypeError('Not supported type for data.' + str(type(data)))