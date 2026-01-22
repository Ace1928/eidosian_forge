import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def require_keyword_args(error: bool) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning or error when passed as a positional argument.

    Modified from sklearn utils.validation.

    Parameters
    ----------
    error :
        Whether to throw an error or raise a warning.
    """

    def throw_if(func: Callable[..., _T]) -> Callable[..., _T]:
        """Throw an error/warning if there are positional arguments after the asterisk.

        Parameters
        ----------
        f :
            function to check arguments on.

        """
        sig = signature(func)
        kwonly_args = []
        all_args = []
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(func)
        def inner_f(*args: Any, **kwargs: Any) -> _T:
            extra_args = len(args) - len(all_args)
            if not all_args and extra_args > 0:
                raise TypeError('Keyword argument is required.')
            if extra_args > 0:
                args_msg = [f'{name}' for name, _ in zip(kwonly_args[:extra_args], args[-extra_args:])]
                msg = 'Pass `{}` as keyword args.'.format(', '.join(args_msg))
                if error:
                    raise TypeError(msg)
                warnings.warn(msg, FutureWarning)
            for k, arg in zip(sig.parameters, args):
                kwargs[k] = arg
            return func(**kwargs)
        return inner_f
    return throw_if