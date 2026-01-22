import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def slowlog_get(self, num: Union[int, None]=None, **kwargs) -> ResponseT:
    """
        Get the entries from the slowlog. If ``num`` is specified, get the
        most recent ``num`` items.

        For more information see https://redis.io/commands/slowlog-get
        """
    from redis.client import NEVER_DECODE
    args = ['SLOWLOG GET']
    if num is not None:
        args.append(num)
    decode_responses = self.get_connection_kwargs().get('decode_responses', False)
    if decode_responses is True:
        kwargs[NEVER_DECODE] = []
    return self.execute_command(*args, **kwargs)