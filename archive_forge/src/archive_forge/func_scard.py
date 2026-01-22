import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def scard(self, name: str) -> Union[Awaitable[int], int]:
    """
        Return the number of elements in set ``name``

        For more information see https://redis.io/commands/scard
        """
    return self.execute_command('SCARD', name)