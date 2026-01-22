from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def extract_dtype(value):
    """
    Extract dtype(s) from the passed `value`.

    Parameters
    ----------
    value : object

    Returns
    -------
    numpy.dtype or pandas.Series of numpy.dtypes
    """
    from modin.pandas.utils import is_scalar
    if hasattr(value, 'dtype'):
        return value.dtype
    elif hasattr(value, 'dtypes'):
        return value.dtypes
    elif is_scalar(value):
        if value is None:
            return pandas.api.types.pandas_dtype(value)
        return pandas.api.types.pandas_dtype(type(value))
    else:
        return np.array(value).dtype