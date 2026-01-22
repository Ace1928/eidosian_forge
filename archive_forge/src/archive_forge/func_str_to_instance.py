import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def str_to_instance(s: str, expected_base_type: Optional[type]=None, args: List[Any]=EMPTY_ARGS, kwargs: Dict[str, Any]=EMPTY_KWARGS, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> Any:
    """Use :func:`~triad.utils.convert.str_to_type` to find a matching type
    and instantiate

    :param s: see :func:`~triad.utils.convert.str_to_type`
    :param expected_base_type: see :func:`~triad.utils.convert.str_to_type`
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :return: the instantiated the object
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    t = str_to_type(s, expected_base_type, global_vars, local_vars)
    return t(*args, **kwargs)