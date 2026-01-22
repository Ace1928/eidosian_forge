import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hincrbyfloat(self, name: str, key: str, amount: float=1.0) -> Union[Awaitable[float], float]:
    """
        Increment the value of ``key`` in hash ``name`` by floating ``amount``

        For more information see https://redis.io/commands/hincrbyfloat
        """
    return self.execute_command('HINCRBYFLOAT', name, key, amount)