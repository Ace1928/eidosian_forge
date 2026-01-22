import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
@contextmanager
def cwd_in_path():
    """Context adding the current working directory to sys.path."""
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        cwd = None
    if not cwd:
        yield
    elif cwd in sys.path:
        yield
    else:
        sys.path.insert(0, cwd)
        try:
            yield cwd
        finally:
            try:
                sys.path.remove(cwd)
            except ValueError:
                pass