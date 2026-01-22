import functools
from typing import Callable, TypeVar
from cvxpy.utilities import scopes
Computes an instance method caches the result.

    A result is stored for each unique combination of arguments and
    keyword arguments. Similar to functools.lru_cache, except this works
    decorator works for instance methods (functools.lru_cache decorates
    functions, not methods; using it on a method leaks memory.)

    This decorator should not be used when there are an unbounded or very
    large number of argument and keyword argument combinations.
     