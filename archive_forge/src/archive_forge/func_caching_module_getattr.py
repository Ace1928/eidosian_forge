import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def caching_module_getattr(cls):
    """
    Helper decorator for implementing module-level ``__getattr__`` as a class.

    This decorator must be used at the module toplevel as follows::

        @caching_module_getattr
        class __getattr__:  # The class *must* be named ``__getattr__``.
            @property  # Only properties are taken into account.
            def name(self): ...

    The ``__getattr__`` class will be replaced by a ``__getattr__``
    function such that trying to access ``name`` on the module will
    resolve the corresponding property (which may be decorated e.g. with
    ``_api.deprecated`` for deprecating module globals).  The properties are
    all implicitly cached.  Moreover, a suitable AttributeError is generated
    and raised if no property with the given name exists.
    """
    assert cls.__name__ == '__getattr__'
    props = {name: prop for name, prop in vars(cls).items() if isinstance(prop, property)}
    instance = cls()

    @functools.cache
    def __getattr__(name):
        if name in props:
            return props[name].__get__(instance)
        raise AttributeError(f'module {cls.__module__!r} has no attribute {name!r}')
    return __getattr__