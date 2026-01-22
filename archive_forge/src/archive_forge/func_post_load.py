from __future__ import annotations
import functools
from typing import Any, Callable, cast
def post_load(fn: Callable[..., Any] | None=None, pass_many: bool=False, pass_original: bool=False) -> Callable[..., Any]:
    """Register a method to invoke after deserializing an object. The method
    receives the deserialized data and returns the processed data.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema`'s :func:`~marshmallow.Schema.load` call.
    If ``pass_many=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before deserializing) will be passed as
    an additional argument to the method.

    .. versionchanged:: 3.0.0
        ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    """
    return set_hook(fn, (POST_LOAD, pass_many), pass_original=pass_original)