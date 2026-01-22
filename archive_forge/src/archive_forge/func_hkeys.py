import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hkeys(self, name: str) -> Union[Awaitable[List], List]:
    """
        Return the list of keys within hash ``name``

        For more information see https://redis.io/commands/hkeys
        """
    return self.execute_command('HKEYS', name)