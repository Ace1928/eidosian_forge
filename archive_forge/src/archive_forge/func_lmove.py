import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lmove(self, first_list: str, second_list: str, src: str='LEFT', dest: str='RIGHT') -> ResponseT:
    """
        Atomically returns and removes the first/last element of a list,
        pushing it as the first/last element on the destination list.
        Returns the element being popped and pushed.

        For more information see https://redis.io/commands/lmove
        """
    params = [first_list, second_list, src, dest]
    return self.execute_command('LMOVE', *params)