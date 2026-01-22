import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hmset(self, name: str, mapping: dict) -> Union[Awaitable[str], str]:
    """
        Set key to value within hash ``name`` for each corresponding
        key and value from the ``mapping`` dict.

        For more information see https://redis.io/commands/hmset
        """
    warnings.warn(f'{self.__class__.__name__}.hmset() is deprecated. Use {self.__class__.__name__}.hset() instead.', DeprecationWarning, stacklevel=2)
    if not mapping:
        raise DataError("'hmset' with 'mapping' of length 0")
    items = []
    for pair in mapping.items():
        items.extend(pair)
    return self.execute_command('HMSET', name, *items)