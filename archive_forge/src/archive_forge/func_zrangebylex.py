import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrangebylex(self, name: KeyT, min: EncodableT, max: EncodableT, start: Union[int, None]=None, num: Union[int, None]=None) -> ResponseT:
    """
        Return the lexicographical range of values from sorted set ``name``
        between ``min`` and ``max``.

        If ``start`` and ``num`` are specified, then return a slice of the
        range.

        For more information see https://redis.io/commands/zrangebylex
        """
    if start is not None and num is None or (num is not None and start is None):
        raise DataError('``start`` and ``num`` must both be specified')
    pieces = ['ZRANGEBYLEX', name, min, max]
    if start is not None and num is not None:
        pieces.extend([b'LIMIT', start, num])
    return self.execute_command(*pieces)