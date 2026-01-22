import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def is_param_defined_in_sub(name: str, sub_has_var_args: bool, sub_has_var_kwargs: bool, sub_sig: inspect.Signature, super_param: inspect.Parameter) -> bool:
    return name in sub_sig.parameters or (super_param.kind == Parameter.VAR_POSITIONAL and sub_has_var_args) or (super_param.kind == Parameter.VAR_KEYWORD and sub_has_var_kwargs) or (super_param.kind == Parameter.POSITIONAL_ONLY and sub_has_var_args) or (super_param.kind == Parameter.POSITIONAL_OR_KEYWORD and sub_has_var_args and sub_has_var_kwargs) or (super_param.kind == Parameter.KEYWORD_ONLY and sub_has_var_kwargs)