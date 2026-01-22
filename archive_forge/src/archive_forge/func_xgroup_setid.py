import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xgroup_setid(self, name: KeyT, groupname: GroupT, id: StreamIdT, entries_read: Optional[int]=None) -> ResponseT:
    """
        Set the consumer group last delivered ID to something else.
        name: name of the stream.
        groupname: name of the consumer group.
        id: ID of the last item in the stream to consider already delivered.

        For more information see https://redis.io/commands/xgroup-setid
        """
    pieces = [name, groupname, id]
    if entries_read is not None:
        pieces.extend(['ENTRIESREAD', entries_read])
    return self.execute_command('XGROUP SETID', *pieces)