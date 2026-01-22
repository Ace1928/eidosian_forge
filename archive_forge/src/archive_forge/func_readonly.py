import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def readonly(self, **kwargs) -> ResponseT:
    """
        Enables read queries for a connection to a Redis Cluster replica node.

        For more information see https://redis.io/commands/readonly
        """
    return self.execute_command('READONLY', **kwargs)