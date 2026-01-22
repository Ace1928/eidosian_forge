import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xrange(self, name: KeyT, min: StreamIdT='-', max: StreamIdT='+', count: Union[int, None]=None) -> ResponseT:
    """
        Read stream values within an interval.

        name: name of the stream.

        start: first stream ID. defaults to '-',
               meaning the earliest available.

        finish: last stream ID. defaults to '+',
                meaning the latest available.

        count: if set, only return this many items, beginning with the
               earliest available.

        For more information see https://redis.io/commands/xrange
        """
    pieces = [min, max]
    if count is not None:
        if not isinstance(count, int) or count < 1:
            raise DataError('XRANGE count must be a positive integer')
        pieces.append(b'COUNT')
        pieces.append(str(count))
    return self.execute_command('XRANGE', name, *pieces)