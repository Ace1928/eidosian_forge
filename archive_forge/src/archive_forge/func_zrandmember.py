import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrandmember(self, key: KeyT, count: int=None, withscores: bool=False) -> ResponseT:
    """
        Return a random element from the sorted set value stored at key.

        ``count`` if the argument is positive, return an array of distinct
        fields. If called with a negative count, the behavior changes and
        the command is allowed to return the same field multiple times.
        In this case, the number of returned fields is the absolute value
        of the specified count.

        ``withscores`` The optional WITHSCORES modifier changes the reply so it
        includes the respective scores of the randomly selected elements from
        the sorted set.

        For more information see https://redis.io/commands/zrandmember
        """
    params = []
    if count is not None:
        params.append(count)
    if withscores:
        params.append('WITHSCORES')
    return self.execute_command('ZRANDMEMBER', key, *params)