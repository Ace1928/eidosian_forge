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
def require_in_instance(self, feature_or_class):
    """A class decorator that requires the dependency is available when the class is
        initialised.  This decorator can be used even if the class does not define an ``__init__``
        method.

        Args:
            feature_or_class (str or Type): the name of the feature that requires these
                dependencies.  If this function is called directly as a decorator (for example
                ``@HAS_X.require_in_instance`` as opposed to
                ``@HAS_X.require_in_instance("my feature")``), then the feature name will be taken
                as the name of the class.

        Returns:
            Callable: a class decorator that ensures that the wrapped feature is present if the
            class is initialised.
        """
    if isinstance(feature_or_class, str):
        feature = feature_or_class

        def decorator(class_):
            wrap_method(class_, '__init__', before=_RequireNow(self, feature))
            return class_
        return decorator
    class_ = feature_or_class
    feature = getattr(class_, '__qualname__', None) or getattr(class_, '__name__', None) or str(class_)
    wrap_method(class_, '__init__', before=_RequireNow(self, feature))
    return class_