import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def create_cache(cache_type, options):
    if cache_type == 'internal':
        from . import internal_cache
        cache = internal_cache.InternalLRUCache(**options)
    elif cache_type == 'sqlite':
        from . import sqlite_cache
        cache = sqlite_cache.SqliteCache(**options)
    elif cache_type == 'redis':
        from . import redis_cache
        cache = redis_cache.RedisCache(**options)
    elif cache_type == 'redis_sentinel':
        from . import redis_cache
        cache = redis_cache.RedisSentinelCache(**options)
    elif cache_type == 'postgres':
        from . import postgres_cache
        cache = postgres_cache.PostgresCache(**options)
    else:
        raise NotImplementedError('Unsupported cache type!')
    return cache