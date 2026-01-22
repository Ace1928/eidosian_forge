import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def canonicalize_to_monomorphic(args: Tuple[Any, ...], kwargs: Dict[Any, Any], default_values: Dict[Any, Any], capture_types: collections.OrderedDict, polymorphic_type: FunctionType) -> Tuple[FunctionType, trace_type.InternalTracingContext]:
    """Generates a monomorphic type out of polymorphic type for given args."""
    poly_bound_arguments = polymorphic_type.bind(*args, **kwargs)
    if default_values:
        poly_bound_arguments.apply_defaults()
        default_values_injected = poly_bound_arguments.arguments
        for name, value in default_values_injected.items():
            if value is CAPTURED_DEFAULT_VALUE:
                default_values_injected[name] = default_values[name]
        poly_bound_arguments = inspect.BoundArguments(poly_bound_arguments.signature, default_values_injected)
    parameters = []
    type_context = trace_type.InternalTracingContext()
    has_var_positional = any((p.kind is Parameter.VAR_POSITIONAL for p in polymorphic_type.parameters.values()))
    for name, arg in poly_bound_arguments.arguments.items():
        poly_parameter = polymorphic_type.parameters[name]
        if has_var_positional and poly_parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            parameters.append(_make_validated_mono_param(name, arg, Parameter.POSITIONAL_ONLY, type_context, poly_parameter.type_constraint))
        elif poly_parameter.kind is Parameter.VAR_POSITIONAL:
            for i, value in enumerate(arg):
                parameters.append(_make_validated_mono_param(f'{poly_parameter.name}_{i}', value, Parameter.POSITIONAL_ONLY, type_context, poly_parameter.type_constraint))
        elif poly_parameter.kind is Parameter.VAR_KEYWORD:
            for kwarg_name in sorted(arg.keys()):
                parameters.append(_make_validated_mono_param(kwarg_name, arg[kwarg_name], Parameter.KEYWORD_ONLY, type_context, poly_parameter.type_constraint))
        else:
            parameters.append(_make_validated_mono_param(name, arg, poly_parameter.kind, type_context, poly_parameter.type_constraint))
    return (FunctionType(parameters, capture_types), type_context)