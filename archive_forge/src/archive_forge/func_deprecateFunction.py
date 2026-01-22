import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
def deprecateFunction(msg, category=UserWarning):
    """Decorator to raise a warning when a deprecated function is called."""

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn('%r is deprecated; %s' % (func.__name__, msg), category=category, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator