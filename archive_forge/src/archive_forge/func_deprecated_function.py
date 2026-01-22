import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Mapping, Union
def deprecated_function(reason='', version='', name=None):
    """
    Decorator to mark a function as deprecated.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(name or func.__name__, reason, version, stacklevel=3)
            return func(*args, **kwargs)
        return wrapper
    return decorator