import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def geoadd(self, name: KeyT, values: Sequence[EncodableT], nx: bool=False, xx: bool=False, ch: bool=False) -> ResponseT:
    """
        Add the specified geospatial items to the specified key identified
        by the ``name`` argument. The Geospatial items are given as ordered
        members of the ``values`` argument, each item or place is formed by
        the triad longitude, latitude and name.

        Note: You can use ZREM to remove elements.

        ``nx`` forces ZADD to only create new elements and not to update
        scores for elements that already exist.

        ``xx`` forces ZADD to only update scores of elements that already
        exist. New elements will not be added.

        ``ch`` modifies the return value to be the numbers of elements changed.
        Changed elements include new elements that were added and elements
        whose scores changed.

        For more information see https://redis.io/commands/geoadd
        """
    if nx and xx:
        raise DataError("GEOADD allows either 'nx' or 'xx', not both")
    if len(values) % 3 != 0:
        raise DataError('GEOADD requires places with lon, lat and name values')
    pieces = [name]
    if nx:
        pieces.append('NX')
    if xx:
        pieces.append('XX')
    if ch:
        pieces.append('CH')
    pieces.extend(values)
    return self.execute_command('GEOADD', *pieces)