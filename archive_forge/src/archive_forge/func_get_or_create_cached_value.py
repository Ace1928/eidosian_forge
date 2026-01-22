from __future__ import annotations
import contextlib
import functools
import hashlib
import inspect
import math
import os
import pickle
import shutil
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterator, TypeVar, cast, overload
from cachetools import TTLCache
import streamlit as st
from streamlit import config, file_util, util
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.elements.spinner import spinner
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.errors import StreamlitAPIWarning
from streamlit.logger import get_logger
from streamlit.runtime.caching import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
from streamlit.runtime.legacy_caching.hashing import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.util import HASHLIB_KWARGS
def get_or_create_cached_value():
    nonlocal cache_key
    if cache_key is None:
        cache_key = _hash_func(non_optional_func, hash_funcs)
    mem_cache = _mem_caches.get_cache(cache_key, max_entries, ttl)
    value_hasher = hashlib.new('md5')
    if args:
        update_hash(args, hasher=value_hasher, hash_funcs=hash_funcs, hash_reason=HashReason.CACHING_FUNC_ARGS, hash_source=non_optional_func)
    if kwargs:
        update_hash(kwargs, hasher=value_hasher, hash_funcs=hash_funcs, hash_reason=HashReason.CACHING_FUNC_ARGS, hash_source=non_optional_func)
    value_key = value_hasher.hexdigest()
    value_key = '{}-{}'.format(value_key, cache_key)
    _LOGGER.debug('Cache key: %s', value_key)
    try:
        return_value = _read_from_cache(mem_cache=mem_cache, key=value_key, persist=persist, allow_output_mutation=allow_output_mutation, func_or_code=non_optional_func, hash_funcs=hash_funcs)
        _LOGGER.debug('Cache hit: %s', non_optional_func)
    except CacheKeyNotFoundError:
        _LOGGER.debug('Cache miss: %s', non_optional_func)
        with _calling_cached_function(non_optional_func):
            if suppress_st_warning:
                with suppress_cached_st_function_warning():
                    return_value = non_optional_func(*args, **kwargs)
            else:
                return_value = non_optional_func(*args, **kwargs)
        _write_to_cache(mem_cache=mem_cache, key=value_key, value=return_value, persist=persist, allow_output_mutation=allow_output_mutation, func_or_code=non_optional_func, hash_funcs=hash_funcs)
    show_deprecation_warning(_make_deprecation_warning(return_value))
    return return_value