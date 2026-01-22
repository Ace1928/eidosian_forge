from typing import Any, Callable, cast, List, Optional
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from contextlib import contextmanager
import importlib
from importlib import abc
import sys
@contextmanager
def delay_import(module_name: str):
    """A context manager that allows the module or submodule named `module_name`
    to be imported without the contents of the module executing until the
    context manager exits.
    """
    delay = True
    execute_list = []

    def wrap_func(module: ModuleType) -> Optional[ModuleType]:
        if delay:
            execute_list.append(module)
            return None
        return module
    with wrap_module_executions(module_name, wrap_func):
        importlib.import_module(module_name)
    yield
    delay = False
    for module in execute_list:
        if module.__loader__ is not None and hasattr(module.__loader__, 'exec_module'):
            cast(Loader, module.__loader__).exec_module(module)