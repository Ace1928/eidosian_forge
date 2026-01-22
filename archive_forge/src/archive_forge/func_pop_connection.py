import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Type
from redis import Connection as RedisConnection
from redis import Redis
from .local import LocalStack
def pop_connection() -> 'Redis':
    """
    Pops the topmost connection from the stack.

    Returns:
        redis (Redis): A Redis connection
    """
    warnings.warn('The `pop_connection` function is deprecated. Pass the `connection` explicitly instead.', DeprecationWarning)
    return _connection_stack.pop()