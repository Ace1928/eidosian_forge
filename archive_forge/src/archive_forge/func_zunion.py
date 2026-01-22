import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zunion(self, keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]], aggregate: Union[str, None]=None, withscores: bool=False) -> ResponseT:
    """
        Return the union of multiple sorted sets specified by ``keys``.
        ``keys`` can be provided as dictionary of keys and their weights.
        Scores will be aggregated based on the ``aggregate``, or SUM if
        none is provided.

        For more information see https://redis.io/commands/zunion
        """
    return self._zaggregate('ZUNION', None, keys, aggregate, withscores=withscores)