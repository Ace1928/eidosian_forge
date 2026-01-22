import contextlib
from functools import wraps
import os
import os.path as osp
import struct
import tempfile
from types import TracebackType
from typing import Any, Callable, TYPE_CHECKING, Optional, Type
from git.types import Literal, PathLike, _T
def post_clear_cache(func: Callable[..., _T]) -> Callable[..., _T]:
    """Decorator for functions that alter the index using the git command. This would
    invalidate our possibly existing entries dictionary which is why it must be
    deleted to allow it to be lazily reread later.

    :note:
        This decorator will not be required once all functions are implemented
        natively which in fact is possible, but probably not feasible performance wise.
    """

    @wraps(func)
    def post_clear_cache_if_not_raised(self: 'IndexFile', *args: Any, **kwargs: Any) -> _T:
        rval = func(self, *args, **kwargs)
        self._delete_entries_cache()
        return rval
    return post_clear_cache_if_not_raised