from __future__ import annotations
import types
from typing import Any
from streamlit import type_util
from streamlit.errors import (
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
def get_return_value_type(return_value: Any) -> str:
    if hasattr(return_value, '__module__') and hasattr(type(return_value), '__name__'):
        return f'`{return_value.__module__}.{type(return_value).__name__}`'
    return get_cached_func_name_md(return_value)