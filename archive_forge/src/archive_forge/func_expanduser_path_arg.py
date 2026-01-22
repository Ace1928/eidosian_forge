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
def expanduser_path_arg(argname: str) -> Callable[[Fn], Fn]:
    """
    Decorate a function replacing its path argument with "user-expanded" value.

    Parameters
    ----------
    argname : str
        Name of the argument which is containing a path to be expanded.

    Returns
    -------
    callable
        Decorator which performs the replacement.
    """

    def decorator(func: Fn) -> Fn:
        signature = inspect.signature(func)
        assert getattr(signature.parameters.get(argname), 'name', None) == argname, f"Function {func} does not take '{argname}' as argument"

        @functools.wraps(func)
        def wrapped(*args: tuple, **kw: dict) -> Any:
            params = signature.bind(*args, **kw)
            if (patharg := params.arguments.get(argname, None)):
                if isinstance(patharg, str) and patharg.startswith('~'):
                    params.arguments[argname] = os.path.expanduser(patharg)
                elif isinstance(patharg, Path):
                    params.arguments[argname] = patharg.expanduser()
                return func(*params.args, **params.kwargs)
            return func(*args, **kw)
        return wrapped
    return decorator