import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method
def require_in_call(self, feature_or_callable):
    """Create a decorator for callables that requires that the dependency is available when the
        decorated function or method is called.

        Args:
            feature_or_callable (str or Callable): the name of the feature that requires these
                dependencies.  If this function is called directly as a decorator (for example
                ``@HAS_X.require_in_call`` as opposed to
                ``@HAS_X.require_in_call("my feature")``), then the feature name will be taken to be
                the function name, or class and method name as appropriate.

        Returns:
            Callable: a decorator that will make its argument require this dependency before it is
            called.
        """
    if isinstance(feature_or_callable, str):
        feature = feature_or_callable

        def decorator(function):

            @functools.wraps(function)
            def out(*args, **kwargs):
                self.require_now(feature)
                return function(*args, **kwargs)
            return out
        return decorator
    function = feature_or_callable
    feature = getattr(function, '__qualname__', None) or getattr(function, '__name__', None) or str(function)

    @functools.wraps(function)
    def out(*args, **kwargs):
        self.require_now(feature)
        return function(*args, **kwargs)
    return out