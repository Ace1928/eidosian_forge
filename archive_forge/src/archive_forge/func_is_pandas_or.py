from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def is_pandas_or(objs: List[Any], obj_type: Any) -> bool:
    """Check whether the input contains at least one ``obj_type`` object and the
    rest are Pandas DataFrames. This function is a utility function for extending
    :func:`~.infer_execution_engine`

    :param objs: the list of objects to check
    :return: whether all objs are of type ``obj_type`` or pandas DataFrame and at
      least one is of type ``obj_type``
    """
    tc = 0
    for obj in objs:
        if not isinstance(obj, pd.DataFrame):
            if isinstance(obj, obj_type):
                tc += 1
            else:
                return False
    return tc > 0