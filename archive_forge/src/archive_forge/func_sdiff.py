import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sdiff(self, keys: List, *args: List) -> Union[Awaitable[list], list]:
    """
        Return the difference of sets specified by ``keys``

        For more information see https://redis.io/commands/sdiff
        """
    args = list_or_args(keys, args)
    return self.execute_command('SDIFF', *args)