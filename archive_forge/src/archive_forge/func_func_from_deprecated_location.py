import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def func_from_deprecated_location(func_name: str, module: str, deprecation_message: str) -> Callable:
    """
    Create a function that decorates a function ``module.func_name`` with a ``FutureWarning``.

    Parameters
    ----------
    func_name : str
        Function name to decorate.
    module : str
        Module where the function is located.
    deprecation_message : str
        Message to print in a future warning.

    Returns
    -------
    callable
    """

    def deprecated_func(*args: tuple[Any], **kwargs: dict[Any, Any]) -> Any:
        """Call deprecated function."""
        func = getattr(importlib.import_module(module), func_name)
        warnings.warn(deprecation_message, FutureWarning)
        return func(*args, **kwargs)
    return deprecated_func