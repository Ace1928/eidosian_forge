import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
def handle_torch_function(public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    overloaded_args = _get_overloaded_args(relevant_args)
    types = tuple(map(type, overloaded_args))
    if _is_torch_function_mode_enabled():
        with _pop_mode_temporarily() as mode:
            result = mode.__torch_function__(public_api, types, args, kwargs)
        if result is not NotImplemented:
            return result
    for overloaded_arg in overloaded_args:
        torch_func_method = overloaded_arg.__torch_function__
        if hasattr(torch_func_method, '__self__') and torch_func_method.__self__ is overloaded_arg and (torch_func_method is not torch._C._disabled_torch_function_impl):
            warnings.warn('Defining your `__torch_function__ as a plain method is deprecated and will be an error in future, please define it as a classmethod.', DeprecationWarning)
        result = torch_func_method(public_api, types, args, kwargs)
        if result is not NotImplemented:
            return result
    func_name = f'{public_api.__module__}.{public_api.__name__}'
    msg = f"no implementation found for '{func_name}' on types that implement __torch_function__: {[type(arg) for arg in overloaded_args]}"
    if _is_torch_function_mode_enabled():
        msg += f' nor in mode {_get_current_function_mode()}'
    raise TypeError(msg)