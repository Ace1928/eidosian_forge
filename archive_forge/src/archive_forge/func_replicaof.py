import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def replicaof(self, *args, **kwargs) -> ResponseT:
    """
        Update the replication settings of a redis replica, on the fly.

        Examples of valid arguments include:

        NO ONE (set no replication)
        host port (set to the host and port of a redis server)

        For more information see  https://redis.io/commands/replicaof
        """
    return self.execute_command('REPLICAOF', *args, **kwargs)