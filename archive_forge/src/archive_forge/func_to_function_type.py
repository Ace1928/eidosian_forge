import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def to_function_type(fullargspec):
    """Generates FunctionType and default values from fullargspec."""
    default_values = _to_default_values(fullargspec)
    parameters = []
    for arg in fullargspec.args:
        arg_name = function_type_lib.sanitize_arg_name(arg)
        parameters.append(function_type_lib.Parameter(arg_name, function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, arg_name in default_values, None))
    if fullargspec.varargs is not None:
        parameters.append(function_type_lib.Parameter(fullargspec.varargs, function_type_lib.Parameter.VAR_POSITIONAL, False, None))
    for kwarg in fullargspec.kwonlyargs:
        parameters.append(function_type_lib.Parameter(function_type_lib.sanitize_arg_name(kwarg), function_type_lib.Parameter.KEYWORD_ONLY, kwarg in default_values, None))
    if fullargspec.varkw is not None:
        parameters.append(function_type_lib.Parameter(fullargspec.varkw, function_type_lib.Parameter.VAR_KEYWORD, False, None))
    return (function_type_lib.FunctionType(parameters), default_values)