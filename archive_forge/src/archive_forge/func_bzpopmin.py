import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bzpopmin(self, keys: KeysT, timeout: TimeoutSecT=0) -> ResponseT:
    """
        ZPOPMIN a value off of the first non-empty sorted set
        named in the ``keys`` list.

        If none of the sorted sets in ``keys`` has a value to ZPOPMIN,
        then block for ``timeout`` seconds, or until a member gets added
        to one of the sorted sets.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/bzpopmin
        """
    if timeout is None:
        timeout = 0
    keys: list[EncodableT] = list_or_args(keys, None)
    keys.append(timeout)
    return self.execute_command('BZPOPMIN', *keys)