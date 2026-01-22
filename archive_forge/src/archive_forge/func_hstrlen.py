import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hstrlen(self, name: str, key: str) -> Union[Awaitable[int], int]:
    """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``

        For more information see https://redis.io/commands/hstrlen
        """
    return self.execute_command('HSTRLEN', name, key)