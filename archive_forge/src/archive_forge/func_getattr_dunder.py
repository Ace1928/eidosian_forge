from typing import List, Callable
import importlib
import warnings
def getattr_dunder(name):
    if name in all:
        warnings.warn(warning_message, RuntimeWarning)
        package = importlib.import_module(new_module)
        return getattr(package, name)
    raise AttributeError(f'Module {new_module!r} has no attribute {name!r}.')