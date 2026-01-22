import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hmget(self, name: str, keys: List, *args: List) -> Union[Awaitable[List], List]:
    """
        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/hmget
        """
    args = list_or_args(keys, args)
    return self.execute_command('HMGET', name, *args)