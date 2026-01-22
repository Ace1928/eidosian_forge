import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def module_loadex(self, path: str, options: Optional[List[str]]=None, args: Optional[List[str]]=None) -> ResponseT:
    """
        Loads a module from a dynamic library at runtime with configuration directives.

        For more information see https://redis.io/commands/module-loadex
        """
    pieces = []
    if options is not None:
        pieces.append('CONFIG')
        pieces.extend(options)
    if args is not None:
        pieces.append('ARGS')
        pieces.extend(args)
    return self.execute_command('MODULE LOADEX', path, *pieces)