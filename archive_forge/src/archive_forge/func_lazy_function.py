from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
def lazy_function(module: str, name: str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """
    func = None

    def _get_function():
        nonlocal func
        if func is None:
            func = getattr(import_module(module), name)
        return func

    class LazyFunctionMeta(type):

        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

    class LazyFunction(metaclass=LazyFunctionMeta):

        def __call__(self, *args, **kwargs):
            nonlocal func
            if func is None:
                func = getattr(import_module(module), name)
            return func(*args, **kwargs)

        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

        def __str__(self):
            return _get_function().__str__()

        def __repr__(self):
            return f"<{__class__.__name__} object at 0x{id(self):x}>: wrapping '{module}.{name}'"
    return LazyFunction()