import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def brpop(self, keys: List, timeout: Optional[int]=0) -> Union[Awaitable[list], list]:
    """
        RPOP a value off of the first non-empty list
        named in the ``keys`` list.

        If none of the lists in ``keys`` has a value to RPOP, then block
        for ``timeout`` seconds, or until a value gets pushed on to one
        of the lists.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/brpop
        """
    if timeout is None:
        timeout = 0
    keys = list_or_args(keys, None)
    keys.append(timeout)
    return self.execute_command('BRPOP', *keys)