import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def dbsize(self, **kwargs) -> ResponseT:
    """
        Returns the number of keys in the current database

        For more information see https://redis.io/commands/dbsize
        """
    return self.execute_command('DBSIZE', **kwargs)