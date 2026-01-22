imported from that module, which is useful when you're changing files deep
import builtins as builtin_mod
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
from warnings import warn
import types
@contextmanager
def replace_import_hook(new_import):
    saved_import = builtin_mod.__import__
    builtin_mod.__import__ = new_import
    try:
        yield
    finally:
        builtin_mod.__import__ = saved_import