import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hrandfield(self, key: str, count: int=None, withvalues: bool=False) -> ResponseT:
    """
        Return a random field from the hash value stored at key.

        count: if the argument is positive, return an array of distinct fields.
        If called with a negative count, the behavior changes and the command
        is allowed to return the same field multiple times. In this case,
        the number of returned fields is the absolute value of the
        specified count.
        withvalues: The optional WITHVALUES modifier changes the reply so it
        includes the respective values of the randomly selected hash fields.

        For more information see https://redis.io/commands/hrandfield
        """
    params = []
    if count is not None:
        params.append(count)
    if withvalues:
        params.append('WITHVALUES')
    return self.execute_command('HRANDFIELD', key, *params)