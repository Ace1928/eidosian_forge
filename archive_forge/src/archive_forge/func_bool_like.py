from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def bool_like(value, name, optional=False, strict=False):
    """
    Convert to bool or raise if not bool_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow bool. If False, allow types that support
        casting to bool.

    Returns
    -------
    converted : bool
        value converted to a bool
    """
    if optional and value is None:
        return value
    extra_text = ' or None' if optional else ''
    if strict:
        if isinstance(value, bool):
            return value
        else:
            raise TypeError(f'{name} must be a bool{extra_text}')
    if hasattr(value, 'squeeze') and callable(value.squeeze):
        value = value.squeeze()
    try:
        return bool(value)
    except Exception:
        raise TypeError('{} must be a bool (or bool-compatible){}'.format(name, extra_text))