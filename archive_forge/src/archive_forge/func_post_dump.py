from __future__ import annotations
import functools
from typing import Any, Callable, cast
def post_dump(fn: Callable[..., Any] | None=None, pass_many: bool=False, pass_original: bool=False) -> Callable[..., Any]:
    """Register a method to invoke after serializing an object. The method
    receives the serialized object and returns the processed object.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema`'s :func:`~marshmallow.Schema.dump` call.
    If ``pass_many=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before serializing) will be passed as
    an additional argument to the method.

    .. versionchanged:: 3.0.0
        ``many`` is always passed as a keyword arguments to the decorated method.
    """
    return set_hook(fn, (POST_DUMP, pass_many), pass_original=pass_original)