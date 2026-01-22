import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn
from .._utils import sequence_to_str
from ._api import WeightsEnum
def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters
    try:
        keyword_only_start_idx = next((idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY))
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None
    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = (args[:keyword_only_start_idx], args[keyword_only_start_idx:])
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(f'Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional parameter(s) is deprecated since 0.13 and may be removed in the future. Please use keyword parameter(s) instead.')
            kwargs.update(keyword_only_kwargs)
        return fn(*args, **kwargs)
    return wrapper