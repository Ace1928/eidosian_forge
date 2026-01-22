import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lolwut(self, *version_numbers: Union[str, float], **kwargs) -> ResponseT:
    """
        Get the Redis version and a piece of generative computer art

        See: https://redis.io/commands/lolwut
        """
    if version_numbers:
        return self.execute_command('LOLWUT VERSION', *version_numbers, **kwargs)
    else:
        return self.execute_command('LOLWUT', **kwargs)