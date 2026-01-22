from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def deprecate_func(*, since: str, additional_msg: str | None=None, pending: bool=False, package_name: str='qiskit', removal_timeline: str='no earlier than 3 months after the release date', is_property: bool=False):
    """Decorator to indicate a function has been deprecated.

    It should be placed beneath other decorators like `@staticmethod` and property decorators.

    When deprecating a class, set this decorator on its `__init__` function.

    Args:
        since: The version the deprecation started at. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update ``since`` to the new version.
        additional_msg: Put here any additional information, such as what to use instead.
            For example, "Instead, use the function ``new_func`` from the module
            ``<my_module>.<my_submodule>``, which is similar but uses GPU acceleration."
        pending: Set to ``True`` if the deprecation is still pending.
        package_name: The PyPI package name, e.g. "qiskit-nature".
        removal_timeline: How soon can this deprecation be removed? Expects a value
            like "no sooner than 6 months after the latest release" or "in release 9.99".
        is_property: If the deprecated function is a `@property`, set this to True so that the
            generated message correctly describes it as such. (This isn't necessary for
            property setters, as their docstring is ignored by Python.)

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        qualname = func.__qualname__
        mod_name = func.__module__
        if is_property:
            deprecated_entity = f'The property ``{mod_name}.{qualname}``'
        elif '.' in qualname:
            if func.__name__ == '__init__':
                cls_name = qualname[:-len('.__init__')]
                deprecated_entity = f'The class ``{mod_name}.{cls_name}``'
            else:
                deprecated_entity = f'The method ``{mod_name}.{qualname}()``'
        else:
            deprecated_entity = f'The function ``{mod_name}.{qualname}()``'
        msg, category = _write_deprecation_msg(deprecated_entity=deprecated_entity, package_name=package_name, since=since, pending=pending, additional_msg=additional_msg, removal_timeline=removal_timeline)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=category, stacklevel=2)
            return func(*args, **kwargs)
        add_deprecation_to_docstring(wrapper, msg, since=since, pending=pending)
        return wrapper
    return decorator