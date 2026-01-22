import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def transform_scipy_sparse(data: DataType, is_csr: bool) -> DataType:
    """Ensure correct data alignment and data type for scipy sparse inputs. Input should
    be either csr or csc matrix.

    """
    from scipy.sparse import csc_matrix, csr_matrix
    if len(data.indices) != len(data.data):
        raise ValueError(f'length mismatch: {len(data.indices)} vs {len(data.data)}')
    indptr, _ = _ensure_np_dtype(data.indptr, data.indptr.dtype)
    indices, _ = _ensure_np_dtype(data.indices, data.indices.dtype)
    values, _ = _ensure_np_dtype(data.data, data.data.dtype)
    if indptr is not data.indptr or indices is not data.indices or values is not data.data:
        if is_csr:
            data = csr_matrix((values, indices, indptr), shape=data.shape)
        else:
            data = csc_matrix((values, indices, indptr), shape=data.shape)
    return data