import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def ensure_signature_is_compatible(super_callable: _WrappedMethod, sub_callable: _WrappedMethod2, is_static: bool=False) -> None:
    """Ensure that the signature of `sub_callable` is compatible with the signature of `super_callable`.

    Guarantees that any call to `super_callable` will work on `sub_callable` by checking the following criteria:

    1. The return type of `sub_callable` is a subtype of the return type of `super_callable`.
    2. All parameters of `super_callable` are present in `sub_callable`, unless `sub_callable`
       declares `*args` or `**kwargs`.
    3. All positional parameters of `super_callable` appear in the same order in `sub_callable`.
    4. All parameters of `super_callable` are a subtype of the corresponding parameters of `sub_callable`.
    5. All required parameters of `sub_callable` are present in `super_callable`, unless `super_callable`
       declares `*args` or `**kwargs`.

    :param super_callable: Function to check compatibility with.
    :param sub_callable: Function to check compatibility of.
    :param is_static: True if staticmethod and should check first argument.
    """
    super_callable = _unbound_func(super_callable)
    sub_callable = _unbound_func(sub_callable)
    try:
        super_sig = inspect.signature(super_callable)
    except ValueError:
        return
    super_type_hints = _get_type_hints(super_callable)
    sub_sig = inspect.signature(sub_callable)
    sub_type_hints = _get_type_hints(sub_callable)
    method_name = sub_callable.__qualname__
    same_main_module = _is_same_module(sub_callable, super_callable)
    if super_type_hints is not None and sub_type_hints is not None:
        ensure_return_type_compatibility(super_type_hints, sub_type_hints, method_name)
        ensure_all_kwargs_defined_in_sub(super_sig, sub_sig, super_type_hints, sub_type_hints, is_static, method_name)
        ensure_all_positional_args_defined_in_sub(super_sig, sub_sig, super_type_hints, sub_type_hints, is_static, same_main_module, method_name)
    ensure_no_extra_args_in_sub(super_sig, sub_sig, is_static, method_name)