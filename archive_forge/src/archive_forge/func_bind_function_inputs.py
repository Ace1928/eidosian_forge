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
def bind_function_inputs(args, kwargs, function_type, default_values):
    """Bind `args` and `kwargs` into a canonicalized signature args, kwargs."""
    sanitized_kwargs = {function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()}
    if len(kwargs) != len(sanitized_kwargs):
        raise ValueError(f'Name collision after sanitization. Please rename tf.function input parameters. Original: {sorted(kwargs.keys())}, Sanitized: {sorted(sanitized_kwargs.keys())}')
    try:
        bound_arguments = function_type.bind_with_defaults(args, sanitized_kwargs, default_values)
    except Exception as e:
        raise TypeError(f'Binding inputs to tf.function failed due to `{e}`. Received args: {args} and kwargs: {sanitized_kwargs} for signature: {function_type}.') from e
    return bound_arguments