import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xinfo_consumers(self, name: KeyT, groupname: GroupT) -> ResponseT:
    """
        Returns general information about the consumers in the group.
        name: name of the stream.
        groupname: name of the consumer group.

        For more information see https://redis.io/commands/xinfo-consumers
        """
    return self.execute_command('XINFO CONSUMERS', name, groupname)