from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def deprecate_arguments(kwarg_map: dict[str, str | None], category: Type[Warning]=DeprecationWarning, *, since: str | None=None):
    """Deprecated. Instead, use `@deprecate_arg`.

    Args:
        kwarg_map: A dictionary of the old argument name to the new name.
        category: Usually either DeprecationWarning or PendingDeprecationWarning.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update `since` to the new version.

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        func_name = func.__qualname__
        old_kwarg_to_msg = {}
        for old_arg, new_arg in kwarg_map.items():
            msg_suffix = 'will in the future be removed.' if new_arg is None else f'replaced with {new_arg}.'
            old_kwarg_to_msg[old_arg] = f'{func_name} keyword argument {old_arg} is deprecated and {msg_suffix}'

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in kwarg_map.items():
                _maybe_warn_and_rename_kwarg(args, kwargs, func_name=func_name, original_func_co_varnames=wrapper.__original_func_co_varnames, old_arg_name=old, new_alias=new, warning_msg=old_kwarg_to_msg[old], category=category, predicate=None)
            return func(*args, **kwargs)
        if hasattr(func, '__original_func_co_varnames'):
            wrapper.__original_func_co_varnames = func.__original_func_co_varnames
        else:
            wrapper.__original_func_co_varnames = func.__code__.co_varnames
            param_kinds = {param.kind for param in inspect.signature(func).parameters.values()}
            if inspect.Parameter.VAR_POSITIONAL in param_kinds:
                raise ValueError('@deprecate_arg cannot be used with functions that take variable *args. Use warnings.warn() directly instead.')
        for msg in old_kwarg_to_msg.values():
            add_deprecation_to_docstring(wrapper, msg, since=since, pending=issubclass(category, PendingDeprecationWarning))
        return wrapper
    return decorator