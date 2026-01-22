import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence, Tuple, Type
from redis.asyncio.client import Redis
from redis.asyncio.connection import (
from redis.commands import AsyncSentinelCommands
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def slave_for(self, service_name: str, redis_class: Type[Redis]=Redis, connection_pool_class: Type[SentinelConnectionPool]=SentinelConnectionPool, **kwargs):
    """
        Returns redis client instance for the ``service_name`` slave(s).

        A SentinelConnectionPool class is used to retrieve the slave's
        address before establishing a new connection.

        By default clients will be a :py:class:`~redis.Redis` instance.
        Specify a different class to the ``redis_class`` argument if you
        desire something different.

        The ``connection_pool_class`` specifies the connection pool to use.
        The SentinelConnectionPool will be used by default.

        All other keyword arguments are merged with any connection_kwargs
        passed to this class and passed to the connection pool as keyword
        arguments to be used to initialize Redis connections.
        """
    kwargs['is_master'] = False
    connection_kwargs = dict(self.connection_kwargs)
    connection_kwargs.update(kwargs)
    connection_pool = connection_pool_class(service_name, self, **connection_kwargs)
    return redis_class.from_pool(connection_pool)