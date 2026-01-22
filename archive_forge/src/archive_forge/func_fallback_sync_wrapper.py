from __future__ import annotations
import time
import anyio
import inspect
import contextlib 
import functools
import hashlib
from lazyops.types.common import UpperStrEnum
from lazyops.utils import timed_cache
from lazyops.utils.helpers import create_background_task, fail_after
from lazyops.utils.lazy import lazy_import
from lazyops.utils.pooler import ThreadPooler
from lazyops.utils.lazy import get_function_name
from .compat import BaseModel, root_validator, get_pyd_dict
from .base import ENOVAL
from typing import Optional, Dict, Any, Callable, List, Union, TypeVar, Type, overload, TYPE_CHECKING
from aiokeydb.utils.logs import logger
from aiokeydb.utils.helpers import afail_after
def fallback_sync_wrapper(func: FT, session: 'KeyDBSession', _kwargs: CachifyKwargs) -> FT:
    """
    [Sync] Handles the fallback wrapper
    """
    _sess_ctx: Optional['KeyDBSession'] = None

    def _get_sess():
        nonlocal _sess_ctx
        if _sess_ctx is None:
            with contextlib.suppress(Exception):
                if session.client.ping():
                    _sess_ctx = session
        return _sess_ctx

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper for cachify
        """
        _sess = _get_sess()
        if _sess is None:
            with contextlib.suppress(Exception):
                return timed_cache(secs=_kwargs.ttl)
            return func(*args, **kwargs)
        return cachify_sync(_sess, _kwargs)(func)(*args, **kwargs)

    def clear(keys: Optional[Union[str, List[str]]]=None, **kwargs) -> Optional[int]:
        """
        Clears the cache
        """
        return _kwargs.clear(keys=keys)

    def num_hits(*args, **kwargs) -> int:
        """
        Returns the number of hits
        """
        return _kwargs.num_hits

    def num_keys(**kwargs) -> int:
        """
        Returns the number of keys
        """
        return _kwargs.num_keys

    def cache_keys(**kwargs) -> List[str]:
        """
        Returns the keys
        """
        return _kwargs.cache_keys

    def cache_values(**kwargs) -> List[Any]:
        """
        Returns the values
        """
        return _kwargs.cache_values

    def cache_items(**kwargs) -> Dict[str, Any]:
        """
        Returns the items
        """
        return _kwargs.cache_items

    def invalidate_key(key: str, **kwargs) -> int:
        """
        Invalidates the cache
        """
        return _kwargs.invalidate_cache(key)

    def cache_timestamps(**kwargs) -> Dict[str, float]:
        """
        Returns the timestamps
        """
        return _kwargs.cache_timestamps

    def cache_keyhits(**kwargs) -> Dict[str, int]:
        """
        Returns the keyhits
        """
        return _kwargs.cache_keyhits

    def cache_policy(**kwargs) -> Dict[str, Union[int, CachePolicy]]:
        """
        Returns the cache policy
        """
        return {'max_size': _kwargs.cache_max_size, 'max_size_policy': _kwargs.cache_max_size_policy}

    def cache_config(**kwargs) -> Dict[str, Any]:
        """
        Returns the cache config
        """
        values = get_pyd_dict(_kwargs, exclude={'session'})
        for k, v in values.items():
            if callable(v):
                values[k] = get_function_name(v)
        return values

    def cache_info(**kwargs) -> Dict[str, Any]:
        """
        Returns the info for the cache
        """
        return _kwargs.cache_info

    def cache_update(**kwargs) -> Dict[str, Any]:
        """
        Updates the cache config
        """
        _kwargs.update(**kwargs)
        return cache_config(**kwargs)
    wrapper.clear = clear
    wrapper.num_hits = num_hits
    wrapper.num_keys = num_keys
    wrapper.cache_keys = cache_keys
    wrapper.cache_values = cache_values
    wrapper.cache_items = cache_items
    wrapper.invalidate_key = invalidate_key
    wrapper.cache_timestamps = cache_timestamps
    wrapper.cache_keyhits = cache_keyhits
    wrapper.cache_policy = cache_policy
    wrapper.cache_config = cache_config
    wrapper.cache_info = cache_info
    wrapper.cache_update = cache_update
    return wrapper