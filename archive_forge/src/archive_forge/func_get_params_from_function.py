import inspect
import sys
from copy import copy
from typing import Any, Callable, Dict, List, Tuple, Type, cast, get_type_hints
from typing_extensions import Annotated
from ._typing import get_args, get_origin
from .models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta
def get_params_from_function(func: Callable[..., Any]) -> Dict[str, ParamMeta]:
    if sys.version_info >= (3, 10):
        signature = inspect.signature(func, eval_str=True)
    else:
        signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = {}
    for param in signature.parameters.values():
        annotation, typer_annotations = _split_annotation_from_typer_annotations(param.annotation)
        if len(typer_annotations) > 1:
            raise MultipleTyperAnnotationsError(param.name)
        default = param.default
        if typer_annotations:
            [parameter_info] = typer_annotations
            if isinstance(param.default, ParameterInfo):
                raise MixedAnnotatedAndDefaultStyleError(argument_name=param.name, annotated_param_type=type(parameter_info), default_param_type=type(param.default))
            parameter_info = copy(parameter_info)
            if isinstance(parameter_info, OptionInfo) and parameter_info.default is not ...:
                parameter_info.param_decls = (cast(str, parameter_info.default), *(parameter_info.param_decls or ()))
                parameter_info.default = ...
            if parameter_info.default is not ...:
                raise AnnotatedParamWithDefaultValueError(param_type=type(parameter_info), argument_name=param.name)
            if param.default is not param.empty:
                parameter_info.default = param.default
            default = parameter_info
        elif param.name in type_hints:
            annotation = type_hints[param.name]
        if isinstance(default, ParameterInfo):
            parameter_info = copy(default)
            if parameter_info.default is ... and parameter_info.default_factory:
                parameter_info.default = parameter_info.default_factory
            elif parameter_info.default_factory:
                raise DefaultFactoryAndDefaultValueError(argument_name=param.name, param_type=type(parameter_info))
            default = parameter_info
        params[param.name] = ParamMeta(name=param.name, default=default, annotation=annotation)
    return params