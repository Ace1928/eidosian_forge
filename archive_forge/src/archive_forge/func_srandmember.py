import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def srandmember(self, name: str, number: Optional[int]=None) -> Union[str, List, None]:
    """
        If ``number`` is None, returns a random member of set ``name``.

        If ``number`` is supplied, returns a list of ``number`` random
        members of set ``name``. Note this is only available when running
        Redis 2.6+.

        For more information see https://redis.io/commands/srandmember
        """
    args = number is not None and [number] or []
    return self.execute_command('SRANDMEMBER', name, *args)