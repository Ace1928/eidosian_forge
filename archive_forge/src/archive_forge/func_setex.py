import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def setex(self, name: KeyT, time: ExpiryT, value: EncodableT) -> ResponseT:
    """
        Set the value of key ``name`` to ``value`` that expires in ``time``
        seconds. ``time`` can be represented by an integer or a Python
        timedelta object.

        For more information see https://redis.io/commands/setex
        """
    if isinstance(time, datetime.timedelta):
        time = int(time.total_seconds())
    return self.execute_command('SETEX', name, time, value)