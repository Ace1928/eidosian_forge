import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def get_full_type_path(obj: Any) -> str:
    """Get the full module path of the type (if `obj` is class or function) or type
    of the instance (if `obj` is an object instance)

    :param obj: a class/function type or an object instance
    :raises TypeError: if `obj` is None, lambda, or neither a class or a function
    :return: full path string
    """
    if obj is not None:
        if inspect.isclass(obj):
            return '{}.{}'.format(obj.__module__, obj.__name__)
        if inspect.isfunction(obj):
            if obj.__name__.startswith('<lambda'):
                raise TypeError("Can't get full path for lambda functions")
            return '{}.{}'.format(obj.__module__, obj.__name__)
        if isinstance(obj, object):
            return '{}.{}'.format(obj.__class__.__module__, obj.__class__.__name__)
    raise TypeError(f'Unable to get type full path from {obj}')