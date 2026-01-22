import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def str_to_object(expr: str, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> Any:
    """Convert string expression to object. The string expression must express
    a type with relative or full path, or express a local or global instance without
    brackets or operators.

    :param expr: string expression, see examples below
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None
    :return: the object

    :raises ValueError: unable to find a matching object

    .. admonition:: Examples

        .. code-block:: python

            class _Mock:
                def __init__(self, x=1):
                    self.x = x

            m = _Mock()
            assert 1 == str_to_object("m.x")
            assert 1 == str_to_object("m2.x", local_vars={"m2": m})
            assert RuntimeError == str_to_object("RuntimeError")
            assert _Mock == str_to_object("_Mock")

    .. note::

        This function is to dynamically load an object from string expression.
        If you write that string expression as python code at the same location, it
        should generate the same result.
    """
    try:
        if any((not p.isidentifier() for p in expr.split('.'))):
            raise ValueError(f'{expr} is invalid')
        _globals, _locals = get_caller_global_local_vars(global_vars, local_vars)
        if '.' not in expr:
            return eval(expr, _globals, _locals)
        parts = expr.split('.')
        v = _locals.get(parts[0], _globals.get(parts[0], None))
        if v is not None and (not isinstance(v, ModuleType)):
            return eval(expr, _globals, _locals)
        root = '.'.join(parts[:-1])
        return getattr(importlib.import_module(root), parts[-1])
    except ValueError:
        raise
    except Exception:
        raise ValueError(expr)