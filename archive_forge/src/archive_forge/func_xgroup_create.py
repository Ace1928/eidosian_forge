import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xgroup_create(self, name: KeyT, groupname: GroupT, id: StreamIdT='$', mkstream: bool=False, entries_read: Optional[int]=None) -> ResponseT:
    """
        Create a new consumer group associated with a stream.
        name: name of the stream.
        groupname: name of the consumer group.
        id: ID of the last item in the stream to consider already delivered.

        For more information see https://redis.io/commands/xgroup-create
        """
    pieces: list[EncodableT] = ['XGROUP CREATE', name, groupname, id]
    if mkstream:
        pieces.append(b'MKSTREAM')
    if entries_read is not None:
        pieces.extend(['ENTRIESREAD', entries_read])
    return self.execute_command(*pieces)