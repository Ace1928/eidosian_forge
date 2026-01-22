import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zremrangebylex(self, name: KeyT, min: EncodableT, max: EncodableT) -> ResponseT:
    """
        Remove all elements in the sorted set ``name`` between the
        lexicographical range specified by ``min`` and ``max``.

        Returns the number of elements removed.

        For more information see https://redis.io/commands/zremrangebylex
        """
    return self.execute_command('ZREMRANGEBYLEX', name, min, max)