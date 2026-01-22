import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def bind_with_defaults(self, args, kwargs, default_values):
    """Returns BoundArguments with default values filled in."""
    bound_arguments = self.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    with_default_args = collections.OrderedDict()
    for name, value in bound_arguments.arguments.items():
        if value is CAPTURED_DEFAULT_VALUE:
            with_default_args[name] = default_values[name]
        else:
            with_default_args[name] = value
    for arg_name in with_default_args:
        constraint = self.parameters[arg_name].type_constraint
        if constraint:
            with_default_args[arg_name] = constraint._cast(with_default_args[arg_name], trace_type.InternalCastContext(allow_specs=True))
    bound_arguments = inspect.BoundArguments(self, with_default_args)
    return bound_arguments