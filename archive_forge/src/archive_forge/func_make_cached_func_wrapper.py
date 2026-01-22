from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
def make_cached_func_wrapper(info: CachedFuncInfo) -> Callable[..., Any]:
    """Create a callable wrapper around a CachedFunctionInfo.

    Calling the wrapper will return the cached value if it's already been
    computed, and will call the underlying function to compute and cache the
    value otherwise.

    The wrapper also has a `clear` function that can be called to clear
    all of the wrapper's cached values.
    """
    cached_func = CachedFunc(info)

    @functools.wraps(info.func)
    def wrapper(*args, **kwargs):
        return cached_func(*args, **kwargs)
    wrapper.clear = cached_func.clear
    return wrapper