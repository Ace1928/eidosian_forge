import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def blmpop(self, timeout: float, numkeys: int, *args: List[str], direction: str, count: Optional[int]=1) -> Optional[list]:
    """
        Pop ``count`` values (default 1) from first non-empty in the list
        of provided key names.

        When all lists are empty this command blocks the connection until another
        client pushes to it or until the timeout, timeout of 0 blocks indefinitely

        For more information see https://redis.io/commands/blmpop
        """
    args = [timeout, numkeys, *args, direction, 'COUNT', count]
    return self.execute_command('BLMPOP', *args)