import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zdiff(self, keys: KeysT, withscores: bool=False) -> ResponseT:
    """
        Returns the difference between the first and all successive input
        sorted sets provided in ``keys``.

        For more information see https://redis.io/commands/zdiff
        """
    pieces = [len(keys), *keys]
    if withscores:
        pieces.append('WITHSCORES')
    return self.execute_command('ZDIFF', *pieces)