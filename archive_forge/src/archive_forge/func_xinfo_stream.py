import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xinfo_stream(self, name: KeyT, full: bool=False) -> ResponseT:
    """
        Returns general information about the stream.
        name: name of the stream.
        full: optional boolean, false by default. Return full summary

        For more information see https://redis.io/commands/xinfo-stream
        """
    pieces = [name]
    options = {}
    if full:
        pieces.append(b'FULL')
        options = {'full': full}
    return self.execute_command('XINFO STREAM', *pieces, **options)