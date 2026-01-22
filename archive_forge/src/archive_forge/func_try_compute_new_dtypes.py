import warnings
from typing import Optional
import numpy as np
import pandas
from pandas.api.types import is_bool_dtype, is_scalar
from modin.error_message import ErrorMessage
from .operator import Operator
def try_compute_new_dtypes(first, second, infer_dtypes=None, result_dtype=None, axis=0, func=None):
    """
    Precompute resulting dtypes of the binary operation if possible.

    The dtypes won't be precomputed if any of the operands doesn't have their dtypes materialized
    or if the second operand type is not supported. Supported types: PandasQueryCompiler, list,
    dict, tuple, np.ndarray.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand of the binary operation.
    second : PandasQueryCompiler, list-like or scalar
        Second operand of the binary operation.
    infer_dtypes : {"common_cast", "try_sample", "bool", None}, default: None
        How dtypes should be infered (see ``Binary.register`` doc for more info).
    result_dtype : np.dtype, optional
        NumPy dtype of the result. If not specified it will be inferred from the `infer_dtypes` parameter.
    axis : int, default: 0
        Axis to perform the binary operation along.
    func : callable(pandas.DataFrame, pandas.DataFrame) -> pandas.DataFrame, optional
        A callable to be used for the "try_sample" method.

    Returns
    -------
    pandas.Series or None
    """
    if infer_dtypes is None and result_dtype is None:
        return None
    try:
        if infer_dtypes == 'bool' or is_bool_dtype(result_dtype):
            dtypes = maybe_build_dtypes_series(first, second, dtype=pandas.api.types.pandas_dtype(bool))
        elif infer_dtypes == 'common_cast':
            dtypes = maybe_compute_dtypes_common_cast(first, second, axis=axis, func=None)
        elif infer_dtypes == 'try_sample':
            if func is None:
                raise ValueError("'func' must be specified if dtypes infering method is 'try_sample'")
            dtypes = maybe_compute_dtypes_common_cast(first, second, axis=axis, func=func)
        else:
            dtypes = None
    except NotImplementedError:
        dtypes = None
    return dtypes