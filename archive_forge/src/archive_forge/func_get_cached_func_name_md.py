from __future__ import annotations
import types
from typing import Any
from streamlit import type_util
from streamlit.errors import (
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
def get_cached_func_name_md(func: Any) -> str:
    """Get markdown representation of the function name."""
    if hasattr(func, '__name__'):
        return '`%s()`' % func.__name__
    elif hasattr(type(func), '__name__'):
        return f'`{type(func).__name__}`'
    return f'`{type(func)}`'