from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
def astype_array_safe(values: ArrayLike, dtype, copy: bool=False, errors: IgnoreRaise='raise') -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    errors_legal_values = ('raise', 'ignore')
    if errors not in errors_legal_values:
        invalid_arg = f"Expected value of kwarg 'errors' to be one of {list(errors_legal_values)}. Supplied value is '{errors}'"
        raise ValueError(invalid_arg)
    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = f"Expected an instance of {dtype.__name__}, but got the class instead. Try instantiating 'dtype'."
        raise TypeError(msg)
    dtype = pandas_dtype(dtype)
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype
    try:
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        if errors == 'ignore':
            new_values = values
        else:
            raise
    return new_values