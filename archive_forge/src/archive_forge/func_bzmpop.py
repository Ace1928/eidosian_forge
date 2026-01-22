import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bzmpop(self, timeout: float, numkeys: int, keys: List[str], min: Optional[bool]=False, max: Optional[bool]=False, count: Optional[int]=1) -> Optional[list]:
    """
        Pop ``count`` values (default 1) off of the first non-empty sorted set
        named in the ``keys`` list.

        If none of the sorted sets in ``keys`` has a value to pop,
        then block for ``timeout`` seconds, or until a member gets added
        to one of the sorted sets.

        If timeout is 0, then block indefinitely.

        For more information see https://redis.io/commands/bzmpop
        """
    args = [timeout, numkeys, *keys]
    if min and max or (not min and (not max)):
        raise DataError('Either min or max, but not both must be set')
    elif min:
        args.append('MIN')
    else:
        args.append('MAX')
    args.extend(['COUNT', count])
    return self.execute_command('BZMPOP', *args)