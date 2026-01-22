import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xdel(self, name: KeyT, *ids: StreamIdT) -> ResponseT:
    """
        Deletes one or more messages from a stream.
        name: name of the stream.
        *ids: message ids to delete.

        For more information see https://redis.io/commands/xdel
        """
    return self.execute_command('XDEL', name, *ids)