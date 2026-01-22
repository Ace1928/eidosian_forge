import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zremrangebyrank(self, name: KeyT, min: int, max: int) -> ResponseT:
    """
        Remove all elements in the sorted set ``name`` with ranks between
        ``min`` and ``max``. Values are 0-based, ordered from smallest score
        to largest. Values can be negative indicating the highest scores.
        Returns the number of elements removed

        For more information see https://redis.io/commands/zremrangebyrank
        """
    return self.execute_command('ZREMRANGEBYRANK', name, min, max)