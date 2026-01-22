from typing import Any, Callable, cast, List, Optional
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from contextlib import contextmanager
import importlib
from importlib import abc
import sys
@contextmanager
def wrap_module_executions(module_name: str, wrap_func: Callable[[ModuleType], Optional[ModuleType]], after_exec: Callable[[ModuleType], None]=lambda m: None, assert_meta_path_unchanged: bool=True):
    """A context manager that hooks python's import machinery within the
    context.

    `wrap_func` is called before executing the module called `module_name` and
    any of its submodules.  The module returned by `wrap_func` will be executed.
    """

    def wrap(finder: Any) -> Any:
        if not hasattr(finder, 'find_spec'):
            return finder
        return InstrumentedFinder(finder, module_name, wrap_func, after_exec)
    new_meta_path = [wrap(finder) for finder in sys.meta_path]
    try:
        orig_meta_path, sys.meta_path = (sys.meta_path, new_meta_path)
        yield
    finally:
        if assert_meta_path_unchanged:
            assert sys.meta_path == new_meta_path
        sys.meta_path = orig_meta_path