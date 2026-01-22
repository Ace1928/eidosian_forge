from __future__ import annotations
import builtins
from collections import (
from collections.abc import (
import contextlib
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import iterable_not_string
@contextlib.contextmanager
def temp_setattr(obj, attr: str, value, condition: bool=True) -> Generator[None, None, None]:
    """
    Temporarily set attribute on an object.

    Parameters
    ----------
    obj : object
        Object whose attribute will be modified.
    attr : str
        Attribute to modify.
    value : Any
        Value to temporarily set attribute to.
    condition : bool, default True
        Whether to set the attribute. Provided in order to not have to
        conditionally use this context manager.

    Yields
    ------
    object : obj with modified attribute.
    """
    if condition:
        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
    try:
        yield obj
    finally:
        if condition:
            setattr(obj, attr, old_value)