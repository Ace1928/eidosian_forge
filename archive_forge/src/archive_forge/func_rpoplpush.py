import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def rpoplpush(self, src: str, dst: str) -> Union[Awaitable[str], str]:
    """
        RPOP a value off of the ``src`` list and atomically LPUSH it
        on to the ``dst`` list.  Returns the value.

        For more information see https://redis.io/commands/rpoplpush
        """
    return self.execute_command('RPOPLPUSH', src, dst)