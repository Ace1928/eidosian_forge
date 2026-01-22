import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zinterstore(self, dest: KeyT, keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]], aggregate: Union[str, None]=None) -> ResponseT:
    """
        Intersect multiple sorted sets specified by ``keys`` into a new
        sorted set, ``dest``. Scores in the destination will be aggregated
        based on the ``aggregate``. This option defaults to SUM, where the
        score of an element is summed across the inputs where it exists.
        When this option is set to either MIN or MAX, the resulting set will
        contain the minimum or maximum score of an element across the inputs
        where it exists.

        For more information see https://redis.io/commands/zinterstore
        """
    return self._zaggregate('ZINTERSTORE', dest, keys, aggregate)