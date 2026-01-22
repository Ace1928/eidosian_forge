import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
def qualname(obj):
    """Return object name."""
    if not hasattr(obj, '__name__') and hasattr(obj, '__class__'):
        obj = obj.__class__
    q = getattr(obj, '__qualname__', None)
    if '.' not in q:
        q = '.'.join((obj.__module__, q))
    return q