import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xlen(self, name: KeyT) -> ResponseT:
    """
        Returns the number of elements in a given stream.

        For more information see https://redis.io/commands/xlen
        """
    return self.execute_command('XLEN', name)