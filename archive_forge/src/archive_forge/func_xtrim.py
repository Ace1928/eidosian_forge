import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xtrim(self, name: KeyT, maxlen: Union[int, None]=None, approximate: bool=True, minid: Union[StreamIdT, None]=None, limit: Union[int, None]=None) -> ResponseT:
    """
        Trims old messages from a stream.
        name: name of the stream.
        maxlen: truncate old stream messages beyond this size
        Can't be specified with minid.
        approximate: actual stream length may be slightly more than maxlen
        minid: the minimum id in the stream to query
        Can't be specified with maxlen.
        limit: specifies the maximum number of entries to retrieve

        For more information see https://redis.io/commands/xtrim
        """
    pieces: list[EncodableT] = []
    if maxlen is not None and minid is not None:
        raise DataError('Only one of ``maxlen`` or ``minid`` may be specified')
    if maxlen is None and minid is None:
        raise DataError('One of ``maxlen`` or ``minid`` must be specified')
    if maxlen is not None:
        pieces.append(b'MAXLEN')
    if minid is not None:
        pieces.append(b'MINID')
    if approximate:
        pieces.append(b'~')
    if maxlen is not None:
        pieces.append(maxlen)
    if minid is not None:
        pieces.append(minid)
    if limit is not None:
        pieces.append(b'LIMIT')
        pieces.append(limit)
    return self.execute_command('XTRIM', name, *pieces)