import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None, score_cast_func: Union[type, Callable]=float) -> Iterator:
    """
        Make an iterator using the ZSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
    cursor = '0'
    while cursor != 0:
        cursor, data = self.zscan(name, cursor=cursor, match=match, count=count, score_cast_func=score_cast_func)
        yield from data