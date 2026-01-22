import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zmscore(self, key: KeyT, members: List[str]) -> ResponseT:
    """
        Returns the scores associated with the specified members
        in the sorted set stored at key.
        ``members`` should be a list of the member name.
        Return type is a list of score.
        If the member does not exist, a None will be returned
        in corresponding position.

        For more information see https://redis.io/commands/zmscore
        """
    if not members:
        raise DataError('ZMSCORE members must be a non-empty list')
    pieces = [key] + members
    return self.execute_command('ZMSCORE', *pieces)