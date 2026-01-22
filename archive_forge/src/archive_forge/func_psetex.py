import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def psetex(self, name: KeyT, time_ms: ExpiryT, value: EncodableT):
    """
        Set the value of key ``name`` to ``value`` that expires in ``time_ms``
        milliseconds. ``time_ms`` can be represented by an integer or a Python
        timedelta object

        For more information see https://redis.io/commands/psetex
        """
    if isinstance(time_ms, datetime.timedelta):
        time_ms = int(time_ms.total_seconds() * 1000)
    return self.execute_command('PSETEX', name, time_ms, value)