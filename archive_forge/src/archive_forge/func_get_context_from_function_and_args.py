import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
def get_context_from_function_and_args(function: ty.Callable, args: ty.List[ty.Any], kwargs: ty.Dict[str, ty.Any]) -> ty.Optional[RequestContext]:
    """Find an arg of type RequestContext and return it.

    This is useful in a couple of decorators where we don't know much about the
    function we're wrapping.
    """
    for arg in itertools.chain(kwargs.values(), args):
        if isinstance(arg, RequestContext):
            return arg
    return None