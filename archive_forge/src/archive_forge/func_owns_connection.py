import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence, Tuple, Type
from redis.asyncio.client import Redis
from redis.asyncio.connection import (
from redis.commands import AsyncSentinelCommands
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def owns_connection(self, connection: Connection):
    check = not self.is_master or (self.is_master and self.master_address == (connection.host, connection.port))
    return check and super().owns_connection(connection)