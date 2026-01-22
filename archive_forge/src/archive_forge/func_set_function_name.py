from __future__ import annotations
import os
import platform
import sys
from typing import TYPE_CHECKING
from pandas.compat._constants import (
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
def set_function_name(f: F, name: str, cls: type) -> F:
    """
    Bind the name/qualname attributes of the function.
    """
    f.__name__ = name
    f.__qualname__ = f'{cls.__name__}.{name}'
    f.__module__ = cls.__module__
    return f