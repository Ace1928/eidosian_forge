from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def int_like(value: Any, name: str, optional: bool=False, strict: bool=False) -> Optional[int]:
    """
    Convert to int or raise if not int_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.

    Returns
    -------
    converted : int
        value converted to a int
    """
    if optional and value is None:
        return None
    is_bool_timedelta = isinstance(value, (bool, np.timedelta64))
    if hasattr(value, 'squeeze') and callable(value.squeeze):
        value = value.squeeze()
    if isinstance(value, (int, np.integer)) and (not is_bool_timedelta):
        return int(value)
    elif not strict and (not is_bool_timedelta):
        try:
            if value == value // 1:
                return int(value)
        except Exception:
            pass
    extra_text = ' or None' if optional else ''
    raise TypeError('{} must be integer_like (int or np.integer, but not bool or timedelta64){}'.format(name, extra_text))