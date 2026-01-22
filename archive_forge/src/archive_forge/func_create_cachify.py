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
@overload
def create_cachify(ttl: Optional[int]=None, keybuilder: Optional[Callable]=None, name: Optional[Union[str, Callable]]=None, typed: Optional[bool]=True, exclude_keys: Optional[List[str]]=None, exclude_null: Optional[bool]=True, exclude_exceptions: Optional[Union[bool, List[Exception]]]=True, exclude_null_values_in_hash: Optional[bool]=None, exclude_default_values_in_hash: Optional[bool]=None, disabled: Optional[Union[bool, Callable]]=None, invalidate_after: Optional[Union[int, Callable]]=None, invalidate_if: Optional[Callable]=None, overwrite_if: Optional[Callable]=None, bypass_if: Optional[Callable]=None, timeout: Optional[float]=1.0, verbose: Optional[bool]=False, super_verbose: Optional[bool]=False, raise_exceptions: Optional[bool]=True, encoder: Optional[Union[str, Callable]]=None, decoder: Optional[Union[str, Callable]]=None, hit_setter: Optional[Callable]=None, hit_getter: Optional[Callable]=None, hset_enabled: Optional[bool]=True, cache_field: Optional[str]=None, session: Optional['KeyDBSession']=None, **kwargs) -> Callable[[FT], FT]:
    """
        Creates a new `cachify` partial decorator with the given kwargs

        Args:
            ttl (Optional[int], optional): The TTL for the cache. Defaults to None.
            keybuilder (Optional[Callable], optional): The keybuilder for the cache. Defaults to None.
            name (Optional[Union[str, Callable]], optional): The name for the cache. Defaults to None.
            typed (Optional[bool], optional): Whether or not to include types in the cache key. Defaults to True.
            exclude_keys (Optional[List[str]], optional): The keys to exclude from the cache key. Defaults to None.
            exclude_null (Optional[bool], optional): Whether or not to exclude null values from the cache. Defaults to True.
            exclude_exceptions (Optional[Union[bool, List[Exception]]], optional): Whether or not to exclude exceptions from the cache. Defaults to True.
            exclude_null_values_in_hash (Optional[bool], optional): Whether or not to exclude null values from the cache hash. Defaults to None.
            exclude_default_values_in_hash (Optional[bool], optional): Whether or not to exclude default values from the cache hash. Defaults to None.
            disabled (Optional[Union[bool, Callable]], optional): Whether or not the cache is disabled. Defaults to None.
            invalidate_after (Optional[Union[int, Callable]], optional): The number of hits after which the cache should be invalidated. Defaults to None.
            invalidate_if (Optional[Callable], optional): The function to determine whether or not the cache should be invalidated. Defaults to None.
            overwrite_if (Optional[Callable], optional): The function to determine whether or not the cache should be overwritten. Defaults to None.
            bypass_if (Optional[Callable], optional): The function to determine whether or not the cache should be bypassed. Defaults to None.
            timeout (Optional[float], optional): The timeout for the cache. Defaults to 1.0.
            verbose (Optional[bool], optional): Whether or not to log verbose messages. Defaults to False.
            super_verbose (Optional[bool], optional): Whether or not to log super verbose messages. Defaults to False.
            raise_exceptions (Optional[bool], optional): Whether or not to raise exceptions. Defaults to True.
            encoder (Optional[Union[str, Callable]], optional): The encoder for the cache. Defaults to None.
            decoder (Optional[Union[str, Callable]], optional): The decoder for the cache. Defaults to None.
            hit_setter (Optional[Callable], optional): The hit setter for the cache. Defaults to None.
            hit_getter (Optional[Callable], optional): The hit getter for the cache. Defaults to None.
            hset_enabled (Optional[bool], optional): Whether or not to use hset/hget/hdel/hmset/hmget/hmgetall. Defaults to True.
            
        """
    ...