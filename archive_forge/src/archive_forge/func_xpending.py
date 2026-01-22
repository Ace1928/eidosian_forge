import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xpending(self, name: KeyT, groupname: GroupT) -> ResponseT:
    """
        Returns information about pending messages of a group.
        name: name of the stream.
        groupname: name of the consumer group.

        For more information see https://redis.io/commands/xpending
        """
    return self.execute_command('XPENDING', name, groupname)