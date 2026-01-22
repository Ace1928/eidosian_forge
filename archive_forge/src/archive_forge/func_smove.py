import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def smove(self, src: str, dst: str, value: str) -> Union[Awaitable[bool], bool]:
    """
        Move ``value`` from set ``src`` to set ``dst`` atomically

        For more information see https://redis.io/commands/smove
        """
    return self.execute_command('SMOVE', src, dst, value)