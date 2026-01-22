import importlib.util
import os
import warnings
from functools import wraps
from typing import Optional
def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.

    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]
    if not missing:

        def decorator(func):
            return func
    else:
        req = f'module: {missing[0]}' if len(missing) == 1 else f'modules: {missing}'

        def decorator(func):

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f'{func.__module__}.{func.__name__} requires {req}')
            return wrapped
    return decorator