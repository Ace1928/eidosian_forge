import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def tfcall(self, lib_name: str, func_name: str, keys: KeysT=None, *args: List) -> ResponseT:
    """
        Invoke a function.

        ``lib_name`` - the library name contains the function.
        ``func_name`` - the function name to run.
        ``keys`` - the keys that will be touched by the function.
        ``args`` - Additional argument to pass to the function.

        For more information see https://redis.io/commands/tfcall/
        """
    return self._tfcall(lib_name, func_name, keys, False, *args)