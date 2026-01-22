from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def required_int_like(value: Any, name: str, strict: bool=False) -> int:
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
    _int = int_like(value, name, optional=False, strict=strict)
    assert _int is not None
    return _int