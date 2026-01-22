import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def assign_params(func, namespace):
    """
    Assign parameters from namespace where func solicits.

    >>> def func(x, y=3):
    ...     print(x, y)
    >>> assigned = assign_params(func, dict(x=2, z=4))
    >>> assigned()
    2 3

    The usual errors are raised if a function doesn't receive
    its required parameters:

    >>> assigned = assign_params(func, dict(y=3, z=4))
    >>> assigned()
    Traceback (most recent call last):
    TypeError: func() ...argument...

    It even works on methods:

    >>> class Handler:
    ...     def meth(self, arg):
    ...         print(arg)
    >>> assign_params(Handler().meth, dict(arg='crystal', foo='clear'))()
    crystal
    """
    sig = inspect.signature(func)
    params = sig.parameters.keys()
    call_ns = {k: namespace[k] for k in params if k in namespace}
    return functools.partial(func, **call_ns)