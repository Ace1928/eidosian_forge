import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hvals(self, name: str) -> Union[Awaitable[List], List]:
    """
        Return the list of values within hash ``name``

        For more information see https://redis.io/commands/hvals
        """
    return self.execute_command('HVALS', name)