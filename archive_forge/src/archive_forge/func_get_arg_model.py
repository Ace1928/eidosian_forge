import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
def get_arg_model(func: Callable, *, exclude: Iterable[str]=tuple(), name: str='ArgModel', strict: bool=True) -> ModelMetaclass:
    """Generate a pydantic model for function arguments.

    func (Callable): The function to generate the schema for.
    exclude (Iterable[str]): Parameter names to ignore.
    name (str): Name of created model class.
    strict (bool): Don't allow extra arguments if no variable keyword arguments
        are allowed on the function.
    RETURNS (ModelMetaclass): A pydantic model.
    """
    sig_args = {}
    try:
        sig = inspect.signature(func)
    except ValueError:
        return create_model(name, __config__=ArgSchemaConfigExtra)
    has_variable = False
    for param in sig.parameters.values():
        if param.name in exclude:
            continue
        if param.kind == param.VAR_KEYWORD:
            has_variable = True
            continue
        annotation = param.annotation if param.annotation != param.empty else Any
        default_empty = None if is_cython_func(func) else ...
        default = param.default if param.default != param.empty else default_empty
        sig_args[param.name] = (annotation, default)
    is_strict = strict and (not has_variable)
    sig_args['__config__'] = ArgSchemaConfig if is_strict else ArgSchemaConfigExtra
    return create_model(name, **sig_args)