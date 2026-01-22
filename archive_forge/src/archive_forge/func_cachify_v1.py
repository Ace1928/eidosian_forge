from __future__ import annotations
import anyio
import asyncio
import traceback
import typing
import logging
import contextlib
import functools
from pydantic.types import ByteSize
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.connection import (
from aiokeydb.v2.core import (
from aiokeydb.v2.typing import Number, KeyT, AbsExpiryT, ExpiryT, PatternT
from aiokeydb.v2.configs import KeyDBSettings, KeyDBWorkerSettings, settings as default_settings
from aiokeydb.v2.types import KeyDBUri
from aiokeydb.v2.types.session import KeyDBSession, ClientPools
from aiokeydb.v2.serializers import SerializerType, BaseSerializer
def cachify_v1(cls, cache_ttl: int=None, typed: bool=False, cache_prefix: str=None, exclude: typing.List[str]=None, exclude_null: typing.Optional[bool]=False, exclude_return_types: typing.Optional[typing.List[type]]=None, exclude_return_objs: typing.Optional[typing.List[typing.Any]]=None, exclude_kwargs: typing.Optional[typing.List[str]]=None, include_cache_hit: typing.Optional[bool]=False, invalidate_cache_key: typing.Optional[str]=None, _no_cache: typing.Optional[bool]=False, _no_cache_kwargs: typing.Optional[typing.List[str]]=None, _no_cache_validator: typing.Optional[typing.Callable]=None, _func_name: typing.Optional[str]=None, _validate_requests: typing.Optional[bool]=True, _exclude_request_headers: typing.Optional[typing.Union[typing.List[str], bool]]=True, _cache_invalidator: typing.Optional[typing.Union[bool, typing.Callable]]=None, _invalidate_after_n_hits: typing.Optional[int]=None, _session: typing.Optional[str]=None, _lazy_init: typing.Optional[bool]=None, _cache_timeout: typing.Optional[float]=5.0, **kwargs):
    """Memoizing cache decorator. Repeated calls with the same arguments
        will look up the result in cache and avoid function evaluation.

        If `_func_name` is set to None (default), the callable name will be determined
        automatically.

        When expire is set to zero, function results will not be set in the
        cache. Store lookups still occur, however. 

        If typed is set to True, function arguments of different types will be
        cached separately. For example, f(3) and f(3.0) will be treated as
        distinct calls with distinct results.

        WARNING: You can pass param `no_cache=True` to the function being wrapped
        (not to the decorator) to avoid cache. This allows you to control cache usage
        from where you call the function. However, if your function accepts
        `**kwargs` and passes them to another function, it is your responsibility
        to remove this param from `kwargs` if you don't want to pass it further. Otherwise,
        you'll get the "unexpected keyword argument" exception.

        The original underlying function is accessible through the __wrapped__
        attribute. This is useful for introspection or for rewrapping the
        function with a different cache.

        Example:

        >>> from kops.clients.keydb import KeyDBClient
        >>> @KeyDBClient.cachify(expire=1)
        ... async def fibonacci(number):
        ...     if number == 0:
        ...         return 0
        ...     elif number == 1:
        ...         return 1
        ...     else:
        ...         return fibonacci(number - 1) + fibonacci(number - 2)
        >>> print(fibonacci(100))
        ... # 354224848179261915075

        An additional `__cache_key__` attribute can be used to generate the
        cache key used for the given arguments.

        >>> key = fibonacci.__cache_key__(100)
        >>> print(cache[key])
        ... # 54224848179261915075

        Remember to call memoize when decorating a callable. If you forget,
        then a TypeError will occur. Note the lack of parenthenses after
        memoize below:

        >>> @KeyDBClient.cachify
        ... async def test():
        ...     pass
        ... # Traceback (most recent call last):
        ... # <...>

        :param str _func_name: name given for callable (default None, automatic)
        :param bool typed: cache different types separately (default False)
        :param int expire: seconds until arguments expire
            (default None, no expiry)
        :param cache_prefix: prefix to add to key
            (default KeyDBClient.cache_prefix | `cache_`)
        :type cache_prefix: str | None
        :param exclude: list of arguments to exclude from cache key
            (default None, no exclusion)
        :type exclude: list | None
        :param exclude_null: exclude arguments with null values from cache key
            (default False)
        :type exclude_null: bool
        :param exclude_return_types: list of return types to exclude from cache
            (default None, no exclusion)
        :type exclude_return_types: list | None
        :param exclude_return_objs: list of return objects to exclude from cache
            (default None, no exclusion)
        :type exclude_return_objs: list | None
        :param exclude_kwargs: list of kwargs to exclude from cache key
            (default None, no exclusion)
        :type exclude_kwargs: list | None
        :param include_cache_hit: include cache hit in return value
            (default False)
        :type include_cache_hit: bool
        :param bool _no_cache: disable cache for this function
            (default False)
        :param list _no_cache_kwargs: list of kwargs to disable cache for
            (default None, no exclusion)
        :param callable _no_cache_validator: callable to validate if cache should be disabled
            (default None, no validation)
        :param bool _validate_requests: validate requests
            (default True)
        :param _exclude_request_headers: list of headers to exclude from request validation
            (default True, exclude all headers)
        :type _exclude_request_headers: list | bool
        :param _cache_invalidator: callable to invalidate cache
            (default None, no invalidation)
        :type _cache_invalidator: callable | bool
        :param str _session: session name
            (default None, use default session)
        :param bool _lazy_init: lazy init session
            (default None, use default session)
        :param float _cache_timeout: timeout of cache operations
            (default 5.0)
        :param kwargs: additional arguments to pass to cache
        
        :return: callable decorator
        """
    if _lazy_init is True:
        return cls.cachify(cache_ttl=cache_ttl, typed=typed, cache_prefix=cache_prefix, exclude=exclude, exclude_null=exclude_null, exclude_return_types=exclude_return_types, exclude_return_objs=exclude_return_objs, exclude_kwargs=exclude_kwargs, include_cache_hit=include_cache_hit, invalidate_cache_key=invalidate_cache_key, _no_cache=_no_cache, _no_cache_kwargs=_no_cache_kwargs, _no_cache_validator=_no_cache_validator, _func_name=_func_name, _validate_requests=_validate_requests, _exclude_request_headers=_exclude_request_headers, _cache_invalidator=_cache_invalidator, _invalidate_after_n_hits=_invalidate_after_n_hits, _cache_timeout=_cache_timeout, **kwargs)
    session = cls.get_session(_session)
    return session.cachify(cache_ttl=cache_ttl, typed=typed, cache_prefix=cache_prefix, exclude=exclude, exclude_null=exclude_null, exclude_return_types=exclude_return_types, exclude_return_objs=exclude_return_objs, exclude_kwargs=exclude_kwargs, include_cache_hit=include_cache_hit, invalidate_cache_key=invalidate_cache_key, _no_cache=_no_cache, _no_cache_kwargs=_no_cache_kwargs, _no_cache_validator=_no_cache_validator, _func_name=_func_name, _validate_requests=_validate_requests, _exclude_request_headers=_exclude_request_headers, _cache_invalidator=_cache_invalidator, _invalidate_after_n_hits=_invalidate_after_n_hits, _cache_timeout=_cache_timeout, **kwargs)