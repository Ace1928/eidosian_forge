import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
def module_file(module):
    """Return the correct original file name of a module."""
    name = module.__file__
    return name[:-1] if name.endswith('.pyc') else name