from __future__ import annotations
import copy
from inspect import Parameter, Signature, signature
from typing import Any, Type, TypeVar
from ..traitlets import HasTraits, Undefined
def signature_has_traits(cls: Type[T]) -> Type[T]:
    """Return a decorated class with a constructor signature that contain Trait names as kwargs."""
    traits = [(name, _get_default(value.default_value)) for name, value in cls.class_traits().items() if not name.startswith('_')]
    old_signature = signature(cls.__init__)
    old_parameter_names = list(old_signature.parameters)
    old_positional_parameters = []
    old_var_positional_parameter = None
    old_keyword_only_parameters = []
    old_var_keyword_parameter = None
    for parameter_name in old_signature.parameters:
        parameter = copy.copy(old_signature.parameters[parameter_name])
        if parameter.kind is Parameter.POSITIONAL_ONLY or parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            old_positional_parameters.append(parameter)
        elif parameter.kind is Parameter.VAR_POSITIONAL:
            old_var_positional_parameter = parameter
        elif parameter.kind is Parameter.KEYWORD_ONLY:
            old_keyword_only_parameters.append(parameter)
        elif parameter.kind is Parameter.VAR_KEYWORD:
            old_var_keyword_parameter = parameter
    if old_var_keyword_parameter is None:
        raise RuntimeError(f'The {cls} constructor does not take **kwargs, which means that the signature can not be expanded with trait names')
    new_parameters = []
    new_parameters += old_positional_parameters[1:]
    if old_var_positional_parameter is not None:
        new_parameters.append(old_var_positional_parameter)
    new_parameters += old_keyword_only_parameters
    new_parameters += [Parameter(name, kind=Parameter.KEYWORD_ONLY, default=default) for name, default in traits if name not in old_parameter_names]
    new_parameters.append(old_var_keyword_parameter)
    cls.__signature__ = Signature(new_parameters)
    return cls