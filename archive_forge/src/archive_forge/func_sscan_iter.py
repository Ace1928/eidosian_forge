import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> Iterator:
    """
        Make an iterator using the SSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
    cursor = '0'
    while cursor != 0:
        cursor, data = self.sscan(name, cursor=cursor, match=match, count=count)
        yield from data