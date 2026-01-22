import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_dump(self) -> Union[Awaitable[str], str]:
    """
        Return the serialized payload of loaded libraries.

        For more information see https://redis.io/commands/function-dump
        """
    from redis.client import NEVER_DECODE
    options = {}
    options[NEVER_DECODE] = []
    return self.execute_command('FUNCTION DUMP', **options)