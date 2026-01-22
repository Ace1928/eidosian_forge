import asyncio
import logging
import logging.config
from typing import List, Tuple, Optional, Union, Any
import functools
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import signal
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from memory_profiler import profile
import cachetools.func
from cachetools import TTLCache
import aiofiles
import aiohttp
from aiohttp import web
import json
from datetime import datetime
from sympy import true
def cache_result(key: str, maxsize: int=256, ttl: int=1200):

    def decorator(func):
        cache = TTLCache(maxsize=maxsize, ttl=ttl)

        @cachetools.func.ttl_cache(cache.maxsize, cache.ttl)
        async def cached_func(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await cached_func(*args, **kwargs)
                logging.debug(f'Cached result for {func.__name__} with key {key}: {result}')
            except Exception as e:
                logging.exception(f'Error in cached function {func.__name__} with args {args} and kwargs {kwargs}: {e}')
                raise e
            return result
        return wrapper
    return decorator