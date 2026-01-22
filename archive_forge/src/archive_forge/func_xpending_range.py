import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xpending_range(self, name: KeyT, groupname: GroupT, min: StreamIdT, max: StreamIdT, count: int, consumername: Union[ConsumerT, None]=None, idle: Union[int, None]=None) -> ResponseT:
    """
        Returns information about pending messages, in a range.

        name: name of the stream.
        groupname: name of the consumer group.
        idle: available from  version 6.2. filter entries by their
        idle-time, given in milliseconds (optional).
        min: minimum stream ID.
        max: maximum stream ID.
        count: number of messages to return
        consumername: name of a consumer to filter by (optional).
        """
    if {min, max, count} == {None}:
        if idle is not None or consumername is not None:
            raise DataError('if XPENDING is provided with idle time or consumername, it must be provided with min, max and count parameters')
        return self.xpending(name, groupname)
    pieces = [name, groupname]
    if min is None or max is None or count is None:
        raise DataError('XPENDING must be provided with min, max and count parameters, or none of them.')
    try:
        if int(idle) < 0:
            raise DataError('XPENDING idle must be a integer >= 0')
        pieces.extend(['IDLE', idle])
    except TypeError:
        pass
    try:
        if int(count) < 0:
            raise DataError('XPENDING count must be a integer >= 0')
        pieces.extend([min, max, count])
    except TypeError:
        pass
    if consumername:
        pieces.append(consumername)
    return self.execute_command('XPENDING', *pieces, parse_detail=True)